import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from bears import FileMetadata
from bears.util import EnvUtil, get_default, ignore_warnings, optional_dependency, retry, set_param_from_alias
from pydantic import confloat, conint, model_validator

from fmcore.framework._task.text_generation import (
    GENERATED_TEXTS_COL,
    GenerativeLM,
    NextTokens,
    Prompts,
    TextGenerationParams,
    TextGenerationParamsMapper,
    TEXT_PROMPT_COL,
)
from fmcore.constants import MLType

# Constants for multimodal support
IMAGE_COL: str = "images"
CONVERSATION_COL: str = "conversations"

# Export classes for public use
__all__ = ["VLLMGenerativeLM", "MultimodalPrompts", "IMAGE_COL", "CONVERSATION_COL"]

"""
VLLM Multimodal Vision-Language Model Support

This module extends the VLLM integration to support Vision-Language Models (VLMs) 
like Qwen 2.5 VL, Qwen 3, LLaVA, and other multimodal models.

Key Features:
- Automatic detection of multimodal models
- Support for image URLs, file paths, and base64 encoded images
- Conversation format compatible with VLLM's multimodal API
- Fallback to text-only mode if multimodal processing fails
- Flexible input handling for both single and multiple images

Usage Examples:

1. Text-only usage (works as before):
```python
from fmcore.algorithm.vllm import VLLMGenerativeLM
from fmcore.framework._task.text_generation import Prompts

# Initialize model
model = VLLMGenerativeLM(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    max_model_len=32768,
    generation_params={"max_new_tokens": 100}
)

# Create text-only prompts
prompts = Prompts.from_data({
    "prompts": ["What is artificial intelligence?"]
})

# Generate responses
predictions = model.predict(prompts)
```

2. Multimodal usage with images:
```python
from fmcore.algorithm.vllm import MultimodalPrompts

# Create multimodal prompts with images
multimodal_prompts = MultimodalPrompts.from_data({
    "prompts": ["Describe this image in detail."],
    "images": ["https://example.com/image.jpg"]  # Can be URL, file path, or list of images
})

# Generate responses with vision understanding
predictions = model.predict(multimodal_prompts)
```

3. Conversation format (most flexible):
```python
conversations = [[
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            {"type": "text", "text": "What do you see in this image?"}
        ]
    }
]]

multimodal_prompts = MultimodalPrompts.from_data({
    "conversations": conversations
})

predictions = model.predict(multimodal_prompts)
```

4. Multiple images:
```python
multimodal_prompts = MultimodalPrompts.from_data({
    "prompts": ["Compare these two images."],
    "images": [["image1.jpg", "image2.jpg"]]  # List of image lists for multiple images per prompt
})
```

Supported Models:
- Qwen/Qwen2.5-VL-* (3B, 7B, 32B, 72B)
- Qwen/Qwen2-VL-*
- llava-hf/llava-*
- internlm/internlm-xcomposer2*
- THUDM/cogvlm*
- Salesforce/blip*
- And other VLLM-compatible vision-language models

Image Input Formats:
- HTTP/HTTPS URLs: "https://example.com/image.jpg"
- File paths: "/path/to/image.jpg" or "image.jpg"
- Base64 data URLs: "data:image/jpeg;base64,..."
- Lists of the above for multiple images

The model automatically detects if it's a multimodal model based on the model name
and handles image processing accordingly. If multimodal processing fails, it falls
back to text-only mode gracefully.
"""

# Define the MultimodalPrompts class outside optional dependency so it's always available
class MultimodalPrompts(Prompts):
    """Extended prompts class that supports both text and images for vision-language models"""
    
    # Allow image columns in the schema
    features_schema = {
        TEXT_PROMPT_COL: MLType.TEXT,  # Keep text prompts
        IMAGE_COL: MLType.IMAGE,  # Add image support
        CONVERSATION_COL: MLType.OBJECT,  # Support conversation format
    }
    
    def to_conversation_format(self) -> List[List[Dict[str, Any]]]:
        """Convert prompts and images to VLLM conversation format"""
        conversations = []
        
        # Check if we have the conversation column directly
        if CONVERSATION_COL in self.data.columns:
            return self.data[CONVERSATION_COL].to_list()
        
        # Otherwise, build conversations from text and image columns
        prompts = self.prompts().to_list()
        images = []
        
        # Get images if they exist
        if IMAGE_COL in self.data.columns:
            images = self.data[IMAGE_COL].to_list()
        
        for i, prompt in enumerate(prompts):
            conversation = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # Add images if available
            if images and i < len(images) and images[i] is not None:
                image_data = images[i]
                # Handle different image input formats
                if isinstance(image_data, str):
                    # Assume it's a URL or base64 string
                    if image_data.startswith(('http://', 'https://', 'data:image')):
                        conversation[0]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": image_data}
                        })
                    else:
                        # Assume it's a file path
                        conversation[0]["content"].append({
                            "type": "image_url", 
                            "image_url": {"url": f"file://{image_data}"}
                        })
                elif isinstance(image_data, list):
                    # Multiple images
                    for img in image_data:
                        if isinstance(img, str):
                            conversation[0]["content"].append({
                                "type": "image_url",
                                "image_url": {"url": img}
                            })
                else:
                    # Assume it's raw image data that needs processing
                    # This would need custom handling based on your image format
                    pass
            
            # Add text prompt
            conversation[0]["content"].append({
                "type": "text", 
                "text": prompt
            })
            
            conversations.append(conversation)
        
        return conversations


with optional_dependency("vllm"):
    from huggingface_hub.errors import HfHubHTTPError
    from vllm import LLM, SamplingParams

    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

    class VLLMGenerativeLM(GenerativeLM):
        aliases = ["vllm", "vllm-vlm", "vllm-multimodal"]

        llm: Optional[LLM] = None
        processor: Optional[Any] = None  # For vision-language models
        cache_dir: Optional[Union[FileMetadata, Dict, str]] = None

        class Hyperparameters(GenerativeLM.Hyperparameters):
            model_name: str
            tensor_parallel_size: Optional[conint(ge=1)] = None
            gpu_memory_utilization: confloat(gt=0.0, le=1.0) = 0.95
            max_model_len: conint(ge=1)
            generation_params: Union[TextGenerationParams, Dict, str]
            api_key: Optional[str] = None
            # New parameters for multimodal support
            is_multimodal: bool = False
            processor_kwargs: Optional[Dict[str, Any]] = None

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
                        "max_length",
                        "max_len",
                        "max_sequence_length",
                        "max_sequence_len",
                        "max_input_length",
                        "max_input_len",
                        "max_model_length",
                        "max_model_len",
                    ],
                )
                set_param_from_alias(
                    params,
                    param="api_key",
                    alias=[
                        "token",
                        "api_token",
                    ],
                )

                # Auto-detect multimodal models
                model_name = params.get("model_name", "")
                if any(keyword in model_name.lower() for keyword in ["qwen2.5-vl", "qwen2-vl", "llava", "internvl", "cogvlm", "blip"]):
                    params["is_multimodal"] = True

                params["generation_params"] = TextGenerationParamsMapper.of(
                    params["generation_params"]
                ).initialize()
                if params.get("cache_dir") is not None:
                    params["cache_dir"] = FileMetadata.of(params["cache_dir"])
                
                # Set default processor kwargs
                if params.get("processor_kwargs") is None:
                    params["processor_kwargs"] = {}
                    
                return params

        def initialize(self, model_dir: Optional[FileMetadata] = None):
            """Initialize the VLLM model and processor"""
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
            kwargs["hf_overrides"]: Dict = dict()
            if self.cache_dir is not None:
                kwargs["download_dir"] = self.cache_dir.path
            if self.hyperparams.api_key is not None:
                kwargs["hf_overrides"]["api_key"] = self.hyperparams.api_key
            print(f"Initializing vllm with kwargs: {kwargs}")

            with ignore_warnings():
                self.llm = retry(
                    LLM,
                    retries=10,
                    wait=30,
                    jitter=0.5,
                    retryable_exceptions=(requests.exceptions.ReadTimeout, HfHubHTTPError),
                    **kwargs,
                )
            
            # Initialize processor for multimodal models
            if self.hyperparams.is_multimodal:
                try:
                    with optional_dependency("transformers"):
                        from transformers import AutoProcessor
                        print(f"Loading processor for multimodal model: {self.hyperparams.model_name}")
                        self.processor = AutoProcessor.from_pretrained(
                            self.hyperparams.model_name,
                            **self.hyperparams.processor_kwargs
                        )
                except Exception as e:
                    print(f"Warning: Could not load processor for multimodal model: {e}")
                    print("Falling back to text-only mode")
                    self.hyperparams.is_multimodal = False

        def predict_step(self, batch: Union[Prompts, MultimodalPrompts], **kwargs) -> Dict:
            """Run prediction on a batch of prompts (text-only or multimodal)"""
            
            # Handle multimodal inputs
            if self.hyperparams.is_multimodal and isinstance(batch, MultimodalPrompts):
                return self._predict_multimodal(batch, **kwargs)
            else:
                return self._predict_text_only(batch, **kwargs)

        def _predict_text_only(self, batch: Prompts, **kwargs) -> Dict:
            """Handle text-only prediction (original functionality)"""
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
                use_tqdm=False,
            )

            return self._process_outputs(outputs)

        def _predict_multimodal(self, batch: MultimodalPrompts, **kwargs) -> Dict:
            """Handle multimodal prediction with images and text"""
            try:
                conversations = batch.to_conversation_format()
                
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

                # For multimodal models, we need to use the conversation format
                # VLLM supports this directly for vision-language models
                outputs = self.llm.generate(
                    conversations,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

                return self._process_outputs(outputs)
                
            except Exception as e:
                print(f"Error in multimodal prediction: {e}")
                print("Falling back to text-only mode")
                # Fallback to text-only if multimodal fails
                return self._predict_text_only(batch, **kwargs)

        def _process_outputs(self, outputs) -> Dict:
            """Process VLLM outputs into the expected format"""
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

        def _create_predictions(self, batch: Union[Prompts, MultimodalPrompts], predictions: Any, **kwargs) -> NextTokens:
            """Convert raw predictions to NextTokens format"""
            return NextTokens.from_task_data(data=batch, predictions=predictions, **kwargs)

        @property
        def max_num_generated_tokens(self) -> int:
            return self.hyperparams.generation_params.max_new_tokens

        def cleanup(self):
            """Cleanup the llm and processor"""
            if self.llm is not None:
                del self.llm
                self.llm = None
            if self.processor is not None:
                del self.processor
                self.processor = None
