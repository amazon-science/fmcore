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


class RegressionData(Dataset):
    tasks = Task.REGRESSION

    ground_truths_schema = {
        "{ground_truth_score_col_name}": MLType.FLOAT,
    }


REGRESSION_PREDICTIONS_FORMAT_MSG: str = """
Regression predictions returned by algorithm must be a column of scores.
This can be a list, tuple, Numpy array, Pandas Series, etc.
""".strip()


class RegressionPredictions(Predictions):
    tasks = Task.REGRESSION

    ground_truths_schema = {
        "{ground_truth_score_col_name}": MLType.FLOAT,
    }

    predictions_schema = {
        "predicted_score": MLType.FLOAT,
    }

    @property
    def predicted_scores(self) -> ScalableSeries:
        predicted_scores_col: str = next(iter(self.data_schema.predictions_schema.keys()))
        return self.data[predicted_scores_col]

    @property
    def ground_truth_scores(self) -> ScalableSeries:
        assert self.has_ground_truths
        ground_truth_scores_col: str = next(iter(self.data_schema.ground_truths_schema.keys()))
        return self.data[ground_truth_scores_col]


class Regressor(Algorithm, ABC):
    tasks = Task.REGRESSION
    inputs = RegressionData
    outputs = RegressionPredictions

    def _create_predictions(
        self, batch: Dataset, predictions: Union[List, np.ndarray], **kwargs
    ) -> RegressionPredictions:
        if not is_list_like(predictions):
            raise ValueError(REGRESSION_PREDICTIONS_FORMAT_MSG)
        predictions: Dict = {"predicted_score": predictions}
        return RegressionPredictions.from_task_data(data=batch, predictions=predictions, **kwargs)
