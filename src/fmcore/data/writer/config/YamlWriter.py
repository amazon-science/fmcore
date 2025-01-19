import io
from typing import *

import yaml

from fmcore.constants import FileFormat
from fmcore.data.writer.config.ConfigWriter import ConfigWriter


class YamlWriter(ConfigWriter):
    file_formats = [FileFormat.YAML]

    def to_str(self, content: Any, **kwargs) -> str:
        stream = io.StringIO()
        yaml.safe_dump(content, stream, **self.filtered_params(yaml.safe_dump))
        return stream.getvalue()
