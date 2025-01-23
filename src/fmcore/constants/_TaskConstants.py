from typing import Union

from autoenum import AutoEnum, auto


class Task(AutoEnum):
    """
    A Task should only relate to the outputs, not the inputs!
    E.g. "Image classification" is not a valid task type, it should just be "classification".
    Within classification, output variation can be made, especially if the predictions and metrics are different.
    E.g. binary, multi-class and multi-label classification can all be considered different tasks since they have
    significantly different metrics.
    """

    ## Classification
    BINARY_CLASSIFICATION = auto()
    MULTI_CLASS_CLASSIFICATION = auto()
    MULTI_LABEL_CLASSIFICATION = auto()

    ## Regression
    REGRESSION = auto()

    ## Embedding
    EMBEDDING = auto()

    NER = auto()

    ## Ranking & Retrieval
    RETRIEVAL_CORPUS = auto()  ## For Datasets
    RANKING = auto()
    RETRIEVAL = auto()

    ## Prompting-based techniques
    NEXT_TOKEN_PREDICTION = auto()  ## Core task
    IN_CONTEXT_LEARNING = auto()  ## Derived task

    ## Audio & Speech
    TEXT_TO_SPEECH = auto()


TaskOrStr = Union[Task, str]
