import json
import json_repair
from typing import Dict

from fmcore.mapper.base_mapper import BaseMapper, I, O


class LLMResponseJsonMapper(BaseMapper[str, Dict]):
    """
    A mapper that processes LLM-generated responses by extracting and repairing JSON content.

    This class is primarily used for parsing responses from LLMs that contain JSON data, converting
    the JSON content into a Python dictionary. It utilizes the 'json_repair' library to handle
    malformed JSON, fixing common formatting issues such as missing quotes or misplaced commas.

    Reference: https://pypi.org/project/json-repair/
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
