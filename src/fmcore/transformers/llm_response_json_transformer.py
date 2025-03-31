import json_repair
from typing import Dict

from fmcore.transformers.base_transformer import BaseTransformer, I, O


class LLMResponseJsonTransformer(BaseTransformer[str, Dict]):
    """
    A transformer that processes LLM-generated responses by extracting and repairing JSON content.

    This class is primarily used for parsing responses from LLMs that contain JSON data, converting
    the JSON content into a Python dictionary. It utilizes the 'json_repair' library to handle
    malformed JSON, fixing common formatting issues such as missing quotes or misplaced commas.

    Reference: https://pypi.org/project/json-repair/
    """

    def transform(self, data: str) -> Dict:
        """
        Parses the LLM's response string, extracting and repairing JSON content as needed.

        Args:
            data (str): The raw response string from the LLM.

        Returns:
            Dict: The extracted and repaired JSON content as a dictionary.
        """
        return json_repair.loads(data)

    async def atransform(self, data: I) -> O:
        """
        Asynchronously parses and repairs JSON content from an LLM response.

        Args:
            data (str): The raw response string from the LLM.

        Returns:
            Dict: The extracted and repaired JSON content as a dictionary.
        """
        return self.transform(data=data)
