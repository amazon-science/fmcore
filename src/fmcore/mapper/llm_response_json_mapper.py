import json_repair
from typing import Dict, List

from fmcore.mapper.base_mapper import BaseMapper


class LLMResponseJsonMapper(BaseMapper[str, List[Dict]]):
    """
    A mapper that processes LLM-generated responses by extracting and repairing JSON content.

    This class is primarily used for parsing responses from LLMs that contain JSON data, converting
    the JSON content into a Python dictionary. It utilizes the 'json_repair' library to handle
    malformed JSON, fixing common formatting issues such as missing quotes or misplaced commas.

    Reference: https://pypi.org/project/json-repair/
    """

    def normalize_json_response(self, json_str: str) -> List[Dict]:
        """
        Normalize and repair JSON response into a list of dictionaries.

        This method takes a JSON string, attempts to repair any malformed JSON using json_repair,
        and ensures the output is always a list of dictionaries. If the input is a single dictionary,
        it will be wrapped in a list. If the input is a list, it will filter out non-dictionary items.

        Args:
            json_str (str): The JSON string to normalize and repair

        Returns:
            List[Dict]: A list of dictionaries containing the normalized JSON data.
                       Returns an empty list if the input cannot be parsed into a dict or list.

        Note:
            This method handles three cases:
            1. Single dictionary -> wraps in list
            2. List of dictionaries -> filters non-dict items
            3. Invalid JSON -> returns empty list
        """
        response = json_repair.loads(json_str)

        if isinstance(response, dict):
            # Wrap dict inside a list
            return [response]

        if isinstance(response, list):
            # Filter list elements, keep only dicts
            filtered = [item for item in response if isinstance(item, dict)]
            return filtered

        # If neither dict nor list, return empty list
        return []

    def map(self, data: str) -> List[Dict]:
        """
        Convert the input string to a list of JSON dictionaries.

        This method processes the input string by attempting to parse it as JSON,
        normalizing the response into a list of dictionaries, and handling any parsing errors.

        Args:
            data (str): The input string containing JSON data to convert

        Returns:
            List[Dict]: A list of dictionaries containing the parsed JSON data

        Raises:
            ValueError: If the JSON parsing fails or the input cannot be converted to valid JSON
        """
        try:
            return self.normalize_json_response(data)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {str(e)}")

    async def amap(self, data: str) -> List[Dict]:
        """
        Asynchronously convert the input string to a JSON dictionary.

        This is an asynchronous wrapper around the map method that provides the same
        functionality but in an async context. Currently, it simply calls the synchronous
        map method as the underlying operations are not I/O bound.

        Args:
            data (str): The input string containing JSON data to convert

        Returns:
            Dict: The parsed JSON dictionary

        Note:
            This method currently delegates to the synchronous map method as the
            underlying JSON parsing operations are CPU-bound rather than I/O-bound.
        """
        return self.map(data)
