from dataclasses import Field
from typing import Dict, List, Optional
from bears.FileMetadata import FileMetadata
from fmcore.types.enums.dataset_enums import DatasetType
from fmcore.types.typed import MutableTyped

class DatasetConfig(MutableTyped):
    inputs: Dict[DatasetType, FileMetadata] = {}
    outputs: FileMetadata
