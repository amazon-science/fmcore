from typing import Dict
import json
import json_repair

from fmcore.mapper.base_mapper import BaseMapper


class LLMResponseJsonMapper(BaseMapper[str, Dict]):
    """
    A mapper that converts LLM response strings to JSON dictionaries.
    """

    def map(self, data: str) -> Dict:
        """
        Convert the input string to a JSON dictionary.

        Args:
            data (str): The input string to convert

        Returns:
            Dict: The parsed JSON dictionary
        """
        try:
            return json.loads(json_repair.repair_json(data))
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {str(e)}")

    async def amap(self, data: str) -> Dict:
        """
        Asynchronously convert the input string to a JSON dictionary.

        Args:
            data (str): The input string to convert

        Returns:
            Dict: The parsed JSON dictionary
        """
        return self.map(data)
