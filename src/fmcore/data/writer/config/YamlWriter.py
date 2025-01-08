from typing import *
import io, yaml
from fmcore.data.writer.config.ConfigWriter import ConfigWriter
from fmcore.util import StructuredBlob
from fmcore.constants import FileFormat, FileContents


class YamlWriter(ConfigWriter):
    file_formats = [FileFormat.YAML]

    def to_str(self, content: Any, **kwargs) -> str:
        stream = io.StringIO()
        yaml.safe_dump(content, stream, **self.filtered_params(yaml.safe_dump))
        return stream.getvalue()
