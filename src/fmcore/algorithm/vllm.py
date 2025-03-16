from typing import Any, Dict, List, Optional, Union

import numpy as np
from bears import FileMetadata
from bears.util import EnvUtil, get_default, optional_dependency, set_param_from_alias
from pydantic import confloat, conint, model_validator

from fmcore.framework._task.text_generation import (
    GENERATED_TEXTS_COL,
    GenerativeLM,
    NextTokens,
    Prompts,
    TextGenerationParams,
    TextGenerationParamsMapper,
)

with optional_dependency("vllm"):
    from vllm import LLM, SamplingParams

    class VLLMGenerativeLM(GenerativeLM):
        aliases = ["vllm"]

        llm: Optional[LLM] = None
        cache_dir: Optional[Union[FileMetadata, Dict, str]] = None

        class Hyperparameters(GenerativeLM.Hyperparameters):
            model_name: str
            tensor_parallel_size: Optional[conint(ge=1)] = None
            gpu_memory_utilization: confloat(gt=0.0, le=1.0) = 0.95
            max_model_len: conint(ge=1)
            generation_params: Union[TextGenerationParams, Dict, str]

            @model_validator(mode="before")
            @classmethod
            def set_params(cls, params: Dict) -> Dict:
                set_param_from_alias(
                    params,
                    param="model_name",
                    alias=[
                        "model",
                        "pretrained_model_name_or_path",
                        "model_name_or_path",
                    ],
                )
                set_param_from_alias(
                    params,
                    param="max_model_len",
                    alias=[
                        "max_len",
                        "max_model_len",
                        "max_sequence_length",
                        "max_sequence_len",
                        "max_input_length",
                        "max_input_len",
                    ],
                )
                params["generation_params"] = TextGenerationParamsMapper.of(
                    params["generation_params"]
                ).initialize()
                if params.get("cache_dir") is not None:
                    params["cache_dir"] = FileMetadata.of(params["cache_dir"])
                return params

        def initialize(self, model_dir: Optional[FileMetadata] = None):
            """Initialize the VLLM model"""
            tensor_parallel_size: Optional[conint(ge=1)] = get_default(
                self.hyperparams.tensor_parallel_size,
                EnvUtil.num_gpus(),  # Use all GPUs by default
            )

            kwargs = dict(
                model=self.hyperparams.model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=self.hyperparams.gpu_memory_utilization,
                max_model_len=self.hyperparams.max_model_len,
            )

            if self.cache_dir is not None:
                kwargs["download_dir"] = self.cache_dir.path

            print(f"Initializing vllm with kwargs: {kwargs}")
            self.llm = LLM(**kwargs)

        def predict_step(self, batch: Prompts, **kwargs) -> Dict:
            """Run prediction on a batch of prompts"""
            prompts: List[str] = batch.prompts().to_list()

            sampling_params = SamplingParams(
                min_tokens=self.hyperparams.generation_params.min_new_tokens,
                max_tokens=self.hyperparams.generation_params.max_new_tokens,
                temperature=0.0
                if not self.hyperparams.generation_params.do_sample
                else self.hyperparams.generation_params.temperature,
                top_p=self.hyperparams.generation_params.top_p
                if hasattr(self.hyperparams.generation_params, "top_p")
                else 1.0,
                top_k=self.hyperparams.generation_params.top_k
                if hasattr(self.hyperparams.generation_params, "top_k")
                else -1,
                stop=self.hyperparams.generation_params.stop_sequences,
                logprobs=self.hyperparams.generation_params.output_scores,
            )
            outputs = self.llm.generate(
                prompts,
                sampling_params=sampling_params,
            )

            result = {GENERATED_TEXTS_COL: [output.outputs[0].text for output in outputs]}

            if self.hyperparams.generation_params.output_scores:
                # Get token IDs and logprobs for each generation
                token_ids = []
                tokens = []
                token_scores = []

                for output in outputs:
                    # Get the first (and only) generation
                    generation = output.outputs[0]

                    # Extract token IDs, tokens and logprobs
                    gen_token_ids = generation.token_ids
                    gen_tokens = generation.tokens
                    gen_logprobs = generation.logprobs

                    # Convert scores based on output_scores_format
                    if self.hyperparams.generation_params.output_scores_format == "probabilities":
                        # Convert from log probabilities to probabilities
                        gen_logprobs = np.exp(gen_logprobs)
                        # Filter based on tolerance
                        if self.hyperparams.generation_params.output_scores_tolerance is not None:
                            mask = gen_logprobs >= self.hyperparams.generation_params.output_scores_tolerance
                            gen_token_ids = [t for t, m in zip(gen_token_ids, mask) if m]
                            gen_tokens = [t for t, m in zip(gen_tokens, mask) if m]
                            gen_logprobs = [s for s, m in zip(gen_logprobs, mask) if m]

                    elif self.hyperparams.generation_params.output_scores_format == "log-probabilities":
                        # Already in log probabilities format
                        # Filter based on tolerance
                        if self.hyperparams.generation_params.output_scores_tolerance is not None:
                            mask = gen_logprobs >= self.hyperparams.generation_params.output_scores_tolerance
                            gen_token_ids = [t for t, m in zip(gen_token_ids, mask) if m]
                            gen_tokens = [t for t, m in zip(gen_tokens, mask) if m]
                            gen_logprobs = [s for s, m in zip(gen_logprobs, mask) if m]

                    elif self.hyperparams.generation_params.output_scores_format == "logits":
                        # Don't filter or modify scores when using raw logits
                        pass

                    token_ids.append(gen_token_ids)
                    tokens.append(gen_tokens)
                    token_scores.append(gen_logprobs)

                result.update(
                    {
                        "generated_token_ids": token_ids,
                        "generated_tokens": tokens,
                        "generated_token_scores": token_scores,
                    }
                )

            return result

        def _create_predictions(self, batch: Prompts, predictions: Any, **kwargs) -> NextTokens:
            """Convert raw predictions to NextTokens format"""
            return NextTokens.from_task_data(data=batch, predictions=predictions, **kwargs)

        @property
        def max_num_generated_tokens(self) -> int:
            return self.hyperparams.generation_params.max_new_tokens

        def cleanup(self):
            """Cleanup the llm"""
            if self.llm is not None:
                del self.llm
                self.llm = None
