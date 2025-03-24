from enum import auto

from autoenum import AutoEnum, alias


class DatasetType(AutoEnum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()
