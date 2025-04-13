from typing import List, Optional
import dspy
from fmcore.llm.base_llm import BaseLLM
from fmcore.llm.types.llm_types import LLMConfig
from langchain_core.messages import BaseMessage

from fmcore.utils.async_utils import AsyncUtils


class DSPyLLMAdapter(dspy.LM):
    """
    Adapter class to interface with a DSPy language model (LM) and an underlying LLM.

    This class allows for inference with either a text prompt or a list of predefined messages.
    It invokes the LLM using DSPy and maintains a history of inputs and outputs for reference.

    Attributes:
        llm: An instance of the `BaseLLM` used for inference.
        history: A list that stores the history of inputs, outputs, and additional parameters.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        **kwargs,
    ):
        """
        Initializes the DSPyLLMAdapter instance with the specified LLM configuration.

        Args:
            llm_config (LLMConfig): Configuration containing model ID and other parameters.
            **kwargs: Additional arguments passed to the base `dspy.LM` class.
        """
        super().__init__(model=llm_config.model_id, **kwargs)
        self.llm: BaseLLM = BaseLLM.of(llm_config=llm_config)
        self.history = []

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[BaseMessage]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Executes inference with either a text prompt or a predefined list of messages.

        If a prompt is provided, it is converted into a list of `HumanMessage` objects.

        Args:
            prompt (str, optional): The input prompt to generate messages for.
            messages (List[BaseMessage], optional): Predefined list of messages for inference.
            **kwargs: Additional keyword arguments passed to the LLM invocation.

        Returns:
            List[str]: The generated responses from the model.

        Raises:
            ValueError: If both prompt and messages are provided.

        Example:
            predictions = model(prompt="The sky is blue")
            print(predictions)
        """
        if prompt and messages:
            raise ValueError("You can only provide either a 'prompt' or 'messages', not both.")

        if prompt:
            messages = [{"role": "user", "content": prompt}]

        response = AsyncUtils.execute(self.llm.ainvoke(messages))
        result = [response.content]

        # Update history with DSPy constructs, which currently support only dictionaries
        entry = {
            "messages": messages,
            "outputs": result,
            "kwargs": kwargs,
        }
        self.history.append(entry)
        self.update_global_history(entry)

        return result


