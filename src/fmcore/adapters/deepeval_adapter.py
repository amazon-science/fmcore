from deepeval.models import DeepEvalBaseLLM
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from fmcore.llm.base_llm import BaseLLM
from langchain_core.messages import BaseMessage


class DeepEvalLLMAdapter(DeepEvalBaseLLM):
    """
    Adapter class that bridges BaseLLM implementations with DeepEval's evaluation framework.

    This adapter implements DeepEval's base LLM interface, allowing any BaseLLM instance
    to be used with DeepEval's evaluation tools. It handles the conversion between
    different message formats and provides both synchronous and asynchronous generation capabilities.

    Note: This adapter currently supports only text generation metrics. Multimodal metrics support will be added soon.

    Attributes:
        llm (BaseLLM): The underlying language model implementation to be adapted.
    """

    llm: BaseLLM

    def __init__(self, llm: BaseLLM):
        """
        Initialize the adapter with a BaseLLM instance.

        Args:
            llm (BaseLLM): The language model implementation to be wrapped.
        """
        self.llm = llm

    def load_model(self):
        """
        Provide access to the underlying LLM instance.

        This method is required by the DeepEval interface to access the model implementation.

        Returns:
            BaseLLM: The wrapped language model instance.
        """
        return self.llm

    def generate(self, prompt: str, schema: BaseModel, **kwargs) -> BaseModel:
        """
        Synchronously generate a response for the given prompt.

        This method converts the string prompt into a response by invoking the underlying
        LLM and extracting the content from the returned message.

        Args:
            prompt (str): The input text to generate a response for.

        Returns:
            str: The generated response text.

        Note:
            This method handles the conversion from DeepEval's string-based interface
            to BaseLLM's message-based interface.
        """
        messages = [HumanMessage(content=prompt)]
        response: BaseMessage = self.llm.invoke(messages=messages)
        return schema.model_validate_json(response.content)

    async def a_generate(self, prompt: str, schema: BaseModel, **kwargs) -> BaseModel:
        """
        Asynchronously generate a response for the given prompt.

        This method provides an asynchronous interface for generating responses,
        useful for high-throughput or I/O-bound evaluation scenarios.

        Args:
            prompt (str): The input text to generate a response for.

        Returns:
            str: The generated response text.

        Note:
            This method handles the conversion from DeepEval's string-based interface
            to BaseLLM's message-based interface in an asynchronous context.
        """
        messages = [HumanMessage(content=prompt)]
        response: BaseMessage = await self.llm.ainvoke(messages=messages)
        return schema.model_validate_json(response.content)

    def get_model_name(self):
        """
        Retrieve the identifier of the underlying language model.

        Returns:
            str: The model identifier as specified in the LLM's configuration.
        """
        return self.llm.config.model_id
