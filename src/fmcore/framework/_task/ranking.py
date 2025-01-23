from abc import ABC
from typing import (
    Dict,
    List,
    Union,
)

import numpy as np
from bears import ScalableSeries
from bears.util import is_list_like

from fmcore.constants import MLType, Task
from fmcore.framework._algorithm import Algorithm
from fmcore.framework._dataset import Dataset
from fmcore.framework._predictions import Predictions
from fmcore.framework._task.embedding import Embeddings


class RankingData(Dataset):
    tasks = Task.RANKING
    ground_truths_schema = {}  ## No ground-truths


RETRIEVAL_FORMAT_MSG: str = """
 Retrieval results returned by algorithm must be a column of vectors.
 """.strip()


class RankedResults(Predictions):
    tasks = Task.RANKING
    ground_truths_schema = {}  ## No ground-truths
    predictions_schema = {
        "embeddings": MLType.VECTOR,
    }

    @property
    def embeddings(self) -> ScalableSeries:
        predicted_embeddings_col: str = next(iter(self.data_schema.predictions_schema.keys()))
        return self.data[predicted_embeddings_col]


class Ranker(Algorithm, ABC):
    tasks = Task.RANKING
    inputs = RankingData
    outputs = RankedResults

    def _create_predictions(
        self, batch: Dataset, predictions: Union[List, np.ndarray], **kwargs
    ) -> Embeddings:
        if not is_list_like(predictions):
            raise ValueError(RETRIEVAL_FORMAT_MSG)
        embeddings: Dict = {"embeddings": predictions}
        return Embeddings.from_task_data(data=batch, predictions=embeddings, **kwargs)
