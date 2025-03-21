import os
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type, Union

from bears import FileMetadata
from bears.util import Log, MappedParameters, String, optional_dependency, retry
from pydantic import confloat, conint, model_validator

from fmcore.framework._task.text_generation import GENERATED_TEXTS_COL, GenerativeLM, Prompts

with optional_dependency("langchain"):
    from langchain import Anthropic, HuggingFaceHub, LLMChain, OpenAI, PromptTemplate
    from langchain.chat_models import ChatOpenAI
    from langchain.llms.base import BaseLLM

    class LangChainLLM(MappedParameters):
        mapping_dict: ClassVar[Dict[str, Type]] = {
            "OpenAI": OpenAI,
            "ChatOpenAI": ChatOpenAI,
            "Anthropic": Anthropic,
        }

    class LangChainPrompter(GenerativeLM):
        aliases = ["LangChain"]

        llm: Optional[BaseLLM] = None

        class Hyperparameters(GenerativeLM.Hyperparameters):
            batch_size: Literal[1] = 1
            llm: Union[LangChainLLM, Dict, str]
            api_key: str
            retries: conint(ge=0) = 3
            retry_wait: confloat(ge=0) = 30.0
            retry_jitter: confloat(ge=0) = 0.25

            @model_validator(mode="before")
            @classmethod
            def set_langchain_params(cls, params: Dict) -> Dict:
                params["batch_size"] = 1
                params["llm"]: LangChainLLM = LangChainLLM.of(params["llm"])
                return params

        @property
        def max_num_generated_tokens(self) -> int:
            if isinstance(self.llm, (OpenAI, ChatOpenAI)):
                return self.llm.max_tokens
            elif isinstance(self.llm, Anthropic):
                return self.llm.max_tokens_to_sample
            elif isinstance(self.llm, HuggingFaceHub):
                return self.llm.model_kwargs["max_new_tokens"]
            raise NotImplementedError(f"Unsupported LangChain LLM:\n{self.hyperparams.llm}")

        def initialize(self, model_dir: Optional[FileMetadata] = None):
            ## Ignore the model_dir.
            if self.hyperparams.llm.mapped_callable() in {OpenAI, ChatOpenAI}:
                os.environ["OPENAI_API_KEY"] = self.hyperparams.api_key
            if self.hyperparams.llm.mapped_callable() == HuggingFaceHub:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.hyperparams.api_key
            self.llm: BaseLLM = self.hyperparams.llm.initialize()

        def predict_step(self, batch: Prompts, **kwargs) -> Any:
            prompt_template: PromptTemplate = PromptTemplate(
                template=batch.prompt_template,
                input_variables=String.str_format_args(batch.prompt_template),
                template_format="f-string",
            )
            llm_chain: LLMChain = LLMChain(prompt=prompt_template, llm=self.llm)
            generated_texts: List = []
            for d in batch.data.to_list_of_dict():
                try:
                    generated_text: str = retry(
                        llm_chain.run,
                        {**d, **dict(return_only_outputs=True)},
                        retries=self.hyperparams.retries,
                        wait=self.hyperparams.retry_wait,
                        jitter=self.hyperparams.retry_jitter,
                        silent=False,
                    )
                except Exception as e:
                    Log.error(String.format_exception_msg(e))
                    generated_text: str = ""
                generated_texts.append(generated_text)
            return {GENERATED_TEXTS_COL: generated_texts}
