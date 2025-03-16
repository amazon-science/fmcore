from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
)

from bears import FileMetadata
from bears.core.frame import ScalableDataFrame
from bears.util import (
    String,
    as_list,
)
from bears.util.aws import S3Util

from fmcore.constants import (
    FILE_FORMAT_TO_FILE_ENDING_MAP,
    FailureAction,
)
from fmcore.framework._dataset import Dataset
from fmcore.framework._evaluator.Evaluator import Evaluator, save_predictions
from fmcore.framework._predictions import Predictions, load_predictions
from fmcore.framework._tracker.Tracker import Tracker
from ._evaluator_utils import algorithm_evaluator_verbosity
from bears.util import DataLoadingStrategy, ShardingStategy


class ParallelAlgorithmEvaluator(ABC):
    """
    Abstract base class for algorithm evaluators that handle distributed evaluation of model predictions.

    This class defines the common interface and shared functionality for evaluators that run
    in distributed environments, whether using Python multiprocessing or Ray.

    Example usage:
        >>> # Implemented by subclasses
    """

    def __init__(self, evaluator: Dict, actor: Tuple[int, int], verbosity: int, **kwargs):
        """
        Initialize the base algorithm evaluator.

        Args:
            evaluator: Dictionary containing evaluator parameters
            actor: Tuple of (actor_index, total_actors)
            verbosity: The verbosity level for logging
            **kwargs: Additional framework-specific initialization parameters
        """
        self.verbosity = verbosity
        self.actor = actor
        ## Set this temporarily while loading when calling Evaluator.of(...):
        self.evaluator: Optional[Evaluator] = None

        ## To be initialized by subclasses
        self._initialize_framework_specific(**kwargs)

        ## Initialize the evaluator with verbosity control
        with algorithm_evaluator_verbosity(self.verbosity):
            self._initialize_evaluator(evaluator)

    @abstractmethod
    def _initialize_framework_specific(self, **kwargs) -> None:
        """
        Initialize framework-specific components.

        Args:
            **kwargs: Framework-specific parameters
        """
        pass

    @abstractmethod
    def _initialize_evaluator(self, evaluator: Dict) -> None:
        """
        Initialize the nested evaluator with the provided parameters.

        Args:
            evaluator: Dictionary containing evaluator parameters
        """
        pass

    @abstractmethod
    def get_evaluator_status(self) -> str:
        """
        Get the current status of the evaluator.

        Returns:
            String representation of the evaluator status
        """
        pass

    @abstractmethod
    def evaluate_shard(
        self,
        data: Any,
        *,
        dataset_params: Dict,
        input_len: int,
        batch_size: int,
        batches_per_save: int,
        predictions_destination: Optional[FileMetadata] = None,
        return_predictions: bool = False,
        failure_action: FailureAction = FailureAction.ERROR,
        **kwargs,
    ):
        """
        Evaluate a shard of data.

        Args:
            data: The data to evaluate
            dataset_params: Parameters for creating the dataset
            input_len: The total number of rows in the dataset
            batch_size: The batch size for model inference
            batches_per_save: Number of batches to process before saving
            predictions_destination: Optional destination for saving predictions
            return_predictions: Whether to return predictions
            failure_action: Action to take on failure
            **kwargs: Additional framework-specific parameters

        Returns:
            Predictions object or None based on return_predictions parameter
        """
        pass

    def _process_batch(
        self,
        macro_batch: ScalableDataFrame,
        dataset_params: Dict,
        batch_size: int,
        macro_batch_save_file: Optional[FileMetadata],
        return_predictions: bool,
        failure_action: FailureAction,
        tracker: Tracker,
        **kwargs,
    ) -> Tuple[Optional[Predictions], bool]:
        """
        Process a single batch of data through the evaluator.

        Args:
            macro_batch: Batch of data to process
            dataset_params: Parameters for dataset creation
            batch_size: Size of individual model batches
            macro_batch_save_file: File to save results to (if any)
            return_predictions: Whether to return predictions
            failure_action: How to handle failures
            tracker: Tracker for logging
            **kwargs: Additional arguments to pass to evaluator

        Returns:
            Tuple of (predictions if requested, whether batch was processed)
        """
        ## Check if we already have results saved
        if macro_batch_save_file is not None and S3Util.s3_object_exists(macro_batch_save_file.path):
            ## Load the file if needed
            if return_predictions:
                predictions = load_predictions(macro_batch_save_file.path)
                return predictions, False
            return None, False

        ## We need to predict
        kwargs["tracker"] = (
            tracker if tracker is not None else Tracker.noop_tracker()
        )  ## Do not track by default
        macro_batch_dataset: Dataset = Dataset.of(
            **dataset_params,
            data=macro_batch,
        )

        with algorithm_evaluator_verbosity(self.verbosity):
            macro_batch_predictions: Predictions = self.evaluator.evaluate(
                macro_batch_dataset,
                batch_size=batch_size,
                return_predictions=True,
                metrics=None,
                progress_bar=None,
                failure_action=FailureAction.ERROR,
                **kwargs,
            )

        if macro_batch_save_file is not None:
            ## Save the predictions:
            save_predictions(
                predictions=macro_batch_predictions,
                predictions_destination=macro_batch_save_file,
            )

        if return_predictions:
            return macro_batch_predictions, True
        return None, True

    def _make_filename_generator(self, dataset_params: Dict, is_sharded: bool):
        """
        Create a filename generator function based on sharding strategy.

        Args:
            dataset_params: Dataset parameters
            is_sharded: Whether data is sharded across actors

        Returns:
            Function to generate filenames for saved predictions
        """
        if is_sharded:
            ## Data is sharded across actors (like in Ray)
            return lambda predicted_num_rows, macro_batch, input_len: (
                f"shard-{String.pad_zeros(*self.actor)}"
                f"-rows-{String.pad_zeros(predicted_num_rows, input_len)}"
                f"-to-{String.pad_zeros(predicted_num_rows + len(macro_batch), input_len)}"
            )
        else:
            ## Data is partitioned (like in ProcessAlgorithmEvaluator)
            return lambda predicted_num_rows, macro_batch, input_len: (
                f"part-{String.pad_zeros(dataset_params['data_idx'] + 1, int(1e9))}"
                f"-rows-{String.pad_zeros(predicted_num_rows, input_len)}"
                f"-to-{String.pad_zeros(predicted_num_rows + len(macro_batch), input_len)}"
            )

    def _get_batch_save_file(
        self,
        predictions_destination: Optional[FileMetadata],
        make_fname: Callable,
        predicted_num_rows: int,
        macro_batch: ScalableDataFrame,
        input_len: int,
    ) -> Optional[FileMetadata]:
        """
        Get the file to save batch predictions to.

        Args:
            predictions_destination: Base destination for predictions
            make_fname: Function to generate filenames
            predicted_num_rows: Number of rows already predicted
            macro_batch: Current batch being processed
            input_len: Total number of rows

        Returns:
            FileMetadata object or None
        """
        if predictions_destination is None:
            return None

        file_ending: str = as_list(FILE_FORMAT_TO_FILE_ENDING_MAP[predictions_destination.format])[0]
        fname: str = make_fname(
            predicted_num_rows=predicted_num_rows,
            macro_batch=macro_batch,
            input_len=input_len,
        )
        return predictions_destination.file_in_dir(
            fname,
            return_metadata=True,
            file_ending=file_ending,
        )
