import json

from fmcore.constants import FileFormat
from fmcore.data.reader.config.ConfigReader import ConfigReader, StructuredBlob


class JsonReader(ConfigReader):
    file_formats = [FileFormat.JSON]

    def _from_str(self, string: str, **kwargs) -> StructuredBlob:
        return json.loads(string, **(self.filtered_params(json.loads)))
