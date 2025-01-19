import json
from typing import *

from fmcore.constants import FileFormat
from fmcore.data.writer.config.ConfigWriter import ConfigWriter
from fmcore.util import StructuredBlob


class JsonWriter(ConfigWriter):
    file_formats = [FileFormat.JSON]

    class Params(ConfigWriter.Params):
        indent: int = 4

    def to_str(self, content: StructuredBlob, **kwargs) -> str:
        return json.dumps(content, **self.filtered_params(json.dumps))
