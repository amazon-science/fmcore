from autoenum import AutoEnum, alias

class DatasetType(AutoEnum):
    TRAIN = alias("train")
    VAL = alias("val", "dev")
    TEST = alias("test")
