from typing import Dict
from bears.FileMetadata import FileMetadata
from fmcore.types.enums.dataset_enums import DatasetType
from fmcore.types.typed import MutableTyped

class DatasetConfig(MutableTyped):
    inputs: Dict[DatasetType, FileMetadata] = {}
    outputs: FileMetadata
