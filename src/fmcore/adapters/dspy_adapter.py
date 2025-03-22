from typing import List, Optional
import dspy
from fmcore.llm.base_llm import BaseLLM
from fmcore.types.llm_types import LLMConfig
from langchain_core.messages import BaseMessage


class DSPyLLMAdapter(dspy.LM):

    def __init__(
        self,
        llm_config: LLMConfig,
        **kwargs,
    ):
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
        Executes inference with either a text prompt or predefined list of messages.

        If a prompt is provided, it is converted into a list of HumanMessage objects.

        Args:
            prompt (str, optional): The input prompt to generate messages for.
            messages (List[BaseMessage], optional): Predefined list of messages for inference.

        Returns:
            List[str]: The generated responses from the model.

        Raises:
            ValueError: If both prompt and messages are provided.

        Example:
            predictions = claude_model(prompt="the sky is blue")
            print(predictions)
        """
        if prompt and messages:
            raise ValueError("You can only provide either a 'prompt' or 'messages', not both.")

        if prompt:
            messages = [{"role": "user", "content": prompt}]

        response = self.llm.invoke(messages)
        result = [response.content]

        # Updating LMs history using DSPy constructs, which currently support only dictionaries
        entry = {
            "messages": messages,
            "outputs": result,
            "kwargs": kwargs,
        }
        self.history.append(entry)
        self.update_global_history(entry)
        return result
