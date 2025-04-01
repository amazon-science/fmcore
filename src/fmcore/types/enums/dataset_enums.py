from enum import auto, Enum


class DatasetType(str, Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"
    VAL = "VAL"

    @classmethod
    def _missing_(cls, value):
        # Normalize the case (convert to uppercase) before comparison
        if isinstance(value, str):
            value = value.upper()
        return super()._missing_(value)
