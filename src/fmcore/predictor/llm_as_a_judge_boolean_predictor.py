from typing import List
from fmcore.llm.base_llm import BaseLLM
from langchain_core.messages import BaseMessage
from fmcore.mapper.llm_response_json_mapper import LLMResponseJsonMapper
from fmcore.mapper.criteria_checker_mapper import CriteriaCheckerMapper

from fmcore.predictor.base_predictor import BasePredictor


class LLMAsAJudgeBooleanPredictor(BasePredictor[bool]):
    """
    A predictor that takes a list of BaseMessages as input and produces a boolean output.
    Sends messages to an LLM and evaluates the response.
    """

    llm: BaseLLM
    json_mapper: LLMResponseJsonMapper
    criteria_checker: CriteriaCheckerMapper

    def predict(self, data: List[BaseMessage]) -> bool:
        """
        Synchronously maps the input messages into a boolean decision.

        Args:
            data (List[BaseMessage]): The input messages to process.

        Returns:
            bool: The predicted boolean output.
        """
        # Get response from LLM
        llm_response = self.llm.invoke(data)
        
        # Parse response to JSON
        json_response = self.json_mapper.map(llm_response.content)
        # Check if response meets criteria
        return self.criteria_checker.map(json_response)

    async def apredict(self, data: List[BaseMessage]) -> bool:
        """
        Asynchronously maps the input messages into a boolean decision.

        Args:
            data (List[BaseMessage]): The input messages to process.

        Returns:
            bool: The predicted boolean output.
        """
        # Get response from LLM
        llm_response = await self.llm.ainvoke(data)
        
        # Parse response to JSON
        json_response = await self.json_mapper.amap(llm_response.content)
        
        # Check if response meets criteria
        return await self.criteria_checker.amap(json_response)