from enum import Enum


class OptimizerType(str, Enum):
    MIPRO_V2 = "MIPRO_V2"


class OptimizerMetricType(str, Enum):
    ACCURACY = "ACCURACY"
