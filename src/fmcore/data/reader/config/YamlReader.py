import io

import yaml

from fmcore.constants import FileFormat
from fmcore.data.reader.config.ConfigReader import ConfigReader, StructuredBlob


class YamlReader(ConfigReader):
    file_formats = [FileFormat.YAML]

    def _from_str(self, string: str, **kwargs) -> StructuredBlob:
        return yaml.safe_load(io.StringIO(string), **(self.filtered_params(yaml.safe_load)))
