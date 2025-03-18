import gc
import math
import multiprocessing as mp
import random
import time
import warnings
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from bears import FileMetadata
from bears.constants import REMOTE_STORAGES, Parallelize
from bears.core.frame import ScalableDataFrame
from bears.util import (
    ActorProxy,
    Alias,
    DataLoadingStrategy,
    LoadBalancingStrategy,
    ProgressBar,
    String,
    Timeout,
    Timer,
    accumulate,
    dispatch_executor,
    get_default,
    get_result,
    is_done,
    safe_validate_arguments,
    set_param_from_alias,
    stop_executor,
    wait,
)
from bears.util.concurrency._processes import actor
from pydantic import ConfigDict, confloat, conint, model_validator

from fmcore.constants import (
    DataLayout,
    FailureAction,
    Storage,
)
from fmcore.framework._dataset import Dataset
from fmcore.framework._evaluator.Evaluator import Evaluator
from fmcore.framework._metric import Metric
from fmcore.framework._predictions import Predictions
from fmcore.framework._tracker.Tracker import Tracker

from .LocalEvaluator import LocalEvaluator
from .ParallelEvaluator import ParallelAlgorithmEvaluator


@actor
class ProcessAlgorithmEvaluator(ParallelAlgorithmEvaluator):
    """
    A process-based algorithm evaluator that handles evaluation of model predictions.

    Example usage:
        >>> evaluator_params = {"evaluator": "local", "task": "generation", "AlgorithmClass": "HFGenerativeLM"}
        >>> actor = ProcessAlgorithmEvaluator.remote(evaluator=evaluator_params, actor=(0, 1), verbosity=2)
        >>> future = actor.evaluate_shard.remote(data, dataset_params=dataset_params, batch_size=4)
    """

    def _init_nested_evaluator(self, evaluator: Dict) -> None:
        """
        Initialize the nested evaluator with the provided parameters.

        Args:
            evaluator: Dictionary containing evaluator parameters
        """
        ## Ensure the nested evaluator is LocalEvaluator
        if evaluator.get("evaluator") not in ["local", LocalEvaluator.class_name]:
            raise ValueError(
                f"{self.class_name} only supports {LocalEvaluator.class_name} as nested evaluator, "
                f"found: {evaluator.get('evaluator')}"
            )

        super()._init_nested_evaluator(evaluator)

    def evaluate_shard(
        self,
        data: Any,
        *,
        dataset_params: Dict,
        input_len: int,
        batch_size: int,
        batches_per_save: int,
        predictions_destination: Optional[FileMetadata],
        return_predictions: bool,
        failure_action: FailureAction,
        data_loading_strategy: DataLoadingStrategy,
        **kwargs,
    ) -> Optional[Predictions]:
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
            data_loading_strategy: Strategy for loading data (only LOCAL supported)
            **kwargs: Additional arguments passed to the evaluator

        Returns:
            Predictions or None depending on return_predictions
        """
        data: ScalableDataFrame = ScalableDataFrame.of(data)
        return self._evaluate_batch_stream(
            data=data,
            dataset_params=dataset_params,
            input_len=input_len,
            batch_size=batch_size,
            batches_per_save=batches_per_save,
            predictions_destination=predictions_destination,
            return_predictions=return_predictions,
            failure_action=failure_action,
            is_sharded=False,  # Process evaluator uses partitioned data
            **kwargs,
        )


class MultiProcessEvaluator(Evaluator):
    """
    An evaluator that uses multiple processes to run predictions in parallel.

    Example usage:
                nested_evaluator="local",
                task="generation",
                AlgorithmClass="HFGenerativeLM",
                num_models=4
            )
        >>> predictions = evaluator.evaluate(dataset, return_predictions=True)
    """

    aliases = ["multiprocess", "mp"]

    model_config = ConfigDict(
        extra="allow",
    )

    nested_evaluator_name: Optional[str] = None
    num_models: Optional[conint(ge=1)] = None
    mp_context: Literal["spawn", "fork", "forkserver"] = "spawn"
    model: Optional[List[Any]] = None  ## Stores the actor proxies
    progress_update_frequency: confloat(ge=0.0) = 15.0
    ## By default, do not cache the model:
    cache_timeout: Optional[Union[Timeout, confloat(gt=0)]] = None

    @model_validator(mode="before")
    @classmethod
    def multiprocess_evaluator_params(cls, params: Dict) -> Dict:
        """Validate and process parameters for the MultiProcessEvaluator."""
        params: Dict = cls._set_common_evaluator_params(params)
        set_param_from_alias(params, param="nested_evaluator_name", alias=["nested_evaluator"])
        Alias.set_max_workers(params, param="num_models")
        set_param_from_alias(
            params,
            param="progress_update_frequency",
            alias=[
                "progress_update_freq",
                "max_report_frequency",
                "progress_update_seconds",
                "progress_update_sec",
            ],
        )
        return params

    def initialize(self, **kwargs):
        """Initialize the evaluator with appropriate defaults."""
        self.nested_evaluator_name: str = get_default(self.nested_evaluator_name, "local")
        if self.nested_evaluator_name != "local" and self.nested_evaluator_name != "LocalEvaluator":
            raise ValueError(
                f"MultiProcessEvaluator only supports 'local' or 'LocalEvaluator' as nested_evaluator_name; "
                f"found: {self.nested_evaluator_name}"
            )

    def _load_model(
        self,
        *,
        num_actors: Optional[int] = None,
        **kwargs,
    ) -> List[Any]:
        """Create the process actors for evaluation."""
        num_actors: int = get_default(num_actors, self.num_actors)
        progress_bar: Optional[Dict] = Alias.get_progress_bar(kwargs)
        progress_bar: Union[Dict, bool] = self._run_evaluation_progress_bar(progress_bar)
        actors_progress_bar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=num_actors,
            desc="Creating Process actors",
            unit="actors",
        )
        nested_evaluator_params: Dict = self._create_nested_evaluator_params(**kwargs)

        ## TODO: fix the spawn creation logic to be faster. Currently, it is super slow
        ## so we have to use a threadpool to create them.
        actor_creation_executor = dispatch_executor(
            parallelize=Parallelize.threads,
            max_workers=min(num_actors, 20),
        )
        actors: List[Any] = []
        for actor_i in range(num_actors):
            actors.append(
                actor_creation_executor.submit(
                    ProcessAlgorithmEvaluator.remote,
                    evaluator=nested_evaluator_params,
                    actor=(actor_i, num_actors),
                    verbosity=self.verbosity,
                    mp_context=self.mp_context,
                )
            )
            time.sleep(0.100)
        for actor_i, actor_future in enumerate(actors):
            actors[actor_i] = actor_future.result()
            actors_progress_bar.update(1)
        stop_executor(actor_creation_executor)
        if len(actors) != num_actors:
            msg: str = f"Creation of {num_actors - len(actors)} actors failed"
            actors_progress_bar.failed(msg)
            raise ValueError(msg)
        else:
            msg: str = f"Created {num_actors} actors"
            actors_progress_bar.success(msg)
        return actors

    def cleanup_model(self):
        """Clean up process actors."""
        self._kill_actors()

    def _kill_actors(self):
        """Kill all process actors and clean up resources."""

        def _stop_actor(actor: ActorProxy):
            actor.stop(cancel_futures=True)
            del actor
            gc.collect()

        try:
            if self.model is not None:
                actors: List[ActorProxy] = self.model
                self.model = None
                ## TODO: fix the spawn stop logic to be faster. Currently, it is super slow
                ## so we have to use a threadpool to stop them.
                actor_stop_executor = dispatch_executor(
                    parallelize=Parallelize.threads,
                    max_workers=min(len(actors), 20),
                )
                accumulate([actor_stop_executor.submit(_stop_actor, actor) for actor in actors])
                stop_executor(actor_stop_executor)
                del actors
        finally:
            gc.collect()

    @property
    def num_actors(self) -> int:
        """Determine the number of actors to use based on settings or defaults."""
        if self.num_models is None:
            warnings.warn(
                f"`num_models` is not specified. Since each model-copy requires "
                f"1 cpu, we create {mp.cpu_count() - 1} model-copies so as "
                f"to utilize the entire machine hardware. "
                f"To reduce the machine-utilization, explicitly pass `num_models`."
            )
            return mp.cpu_count() - 1
        return self.num_models

    def _create_nested_evaluator_params(self, **kwargs) -> Dict:
        """Create parameters for the nested evaluator."""

        if self.model_dir is not None and not self.model_dir.is_remote_storage():
            raise ValueError(
                f"When passing `model_dir` to {self.class_name}.of(...), the model directory "
                f"must be on a remote storage, i.e. one of: {REMOTE_STORAGES}"
            )

        nested_evaluator_name: str = self.nested_evaluator_name
        if "cache_dir" in kwargs and kwargs["cache_dir"] is None:
            kwargs.pop("cache_dir")
        if "model_dir" in kwargs and kwargs["model_dir"] is None:
            kwargs.pop("model_dir")

        nested_evaluator_params: Dict = Evaluator.of(
            **{
                **dict(
                    evaluator=nested_evaluator_name,
                    task=self.task,
                    AlgorithmClass=self.AlgorithmClass,
                    hyperparams=self.hyperparams,
                    model_dir=self.model_dir,
                    cache_dir=self.cache_dir,
                    cache_timeout=self.cache_timeout,
                    validate_inputs=self.validate_inputs,
                    validate_outputs=self.validate_outputs,
                    custom_definitions=self.custom_definitions,
                ),
                **kwargs,
                **dict(
                    ## Since this call is just to create the nested evaluator params,
                    ## we don't need to initialize it:
                    init=False,
                    init_model=False,
                    ## Ensures we do not print anything from the nested evaluator:
                    verbosity=0,
                ),
            }
        ).dict()

        if "stats" in kwargs:
            nested_evaluator_params["stats"] = kwargs["stats"]
        nested_evaluator_params["evaluator"]: str = nested_evaluator_name
        return nested_evaluator_params

    @staticmethod
    def mp_logger(text: str, should_log: bool, tracker: Tracker):
        """Log messages with appropriate verbosity control."""
        text: str = f"{text}"
        if should_log is False:  ## Don't log anything.
            return
        else:
            tracker.info(text)

    @safe_validate_arguments
    def _run_evaluation(
        self,
        dataset: Dataset,
        *,
        tracker: Tracker,
        metrics: Optional[List[Metric]],
        return_predictions: bool,
        predictions_destination: Optional[FileMetadata],
        progress_bar: Optional[Dict],
        failure_action: FailureAction = FailureAction.ERROR_DELAYED,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_USED,
        read_as: Optional[DataLayout] = DataLayout.PANDAS,
        submission_batch_size: Optional[conint(ge=1)] = None,
        worker_queue_len: conint(ge=0) = 2,
        submission_batch_wait: confloat(ge=0) = 3.0,
        submission_batch_wait_jitter: confloat(ge=0.0, le=1.0) = 0.05,
        evaluation_timeout: confloat(ge=0, allow_inf_nan=True) = math.inf,
        allow_partial_predictions: bool = False,
        **kwargs,
    ) -> Tuple[Optional[Predictions], Optional[List[Metric]]]:
        """
        Run evaluation across multiple process actors.

        Only supports local data loading strategy.
        """
        ## Process kwargs and set defaults
        set_param_from_alias(
            kwargs,
            param="batches_per_save",
            alias=["batches_per_update"],
            default=1,
        )
        set_param_from_alias(
            kwargs,
            param="batch_size",
            alias=["predict_batch_size", "eval_batch_size", "nrows", "num_rows"],
            default=self._create_hyperparams().batch_size,
        )
        batch_size: Optional[int] = kwargs.pop("batch_size", None)
        if batch_size is None:
            raise ValueError(
                f"Could not find batch_size in model hyperparams; "
                f"please pass it explicitly like so: {self.class_name}.evaluate(batch_size=...)"
            )
        ## Set submission_batch_size if not already specified
        if submission_batch_size is None:
            submission_batch_size: int = batch_size * kwargs["batches_per_save"]  ## Heuristic

        if predictions_destination is not None:
            if predictions_destination.storage is not Storage.S3:
                raise ValueError(
                    f"Results can only be saved to {Storage.S3}; "
                    f"found storage {predictions_destination.storage} having path: {predictions_destination.path}"
                )
            if not predictions_destination.is_path_valid_dir():
                raise ValueError(
                    f"Expected predictions destination to be a valid directory; "
                    f'found: "{predictions_destination.path}"...did you forget a "/" at the end?'
                )
            assert predictions_destination.format is not None  ## Checked in .evaluate().

        timer: Timer = Timer(silent=True)
        timer.start()
        ## Verbosity >= 1: progress bars
        progress_bar: Union[Dict, bool] = self._run_evaluation_progress_bar(progress_bar)
        ## Verbosity >= 2: basic logging
        main_logger: Callable = partial(
            self.mp_logger,
            ## Unless we request silence (verbosity=0), print important information.
            should_log=self.verbosity >= 2,
            tracker=tracker,
        )
        ## Verbosity >= 3: detailed logging
        debug_logger: Callable = partial(
            self.mp_logger,
            ## Unless we request silence (verbosity=0), print important information.
            should_log=self.verbosity >= 3,
            tracker=tracker,
        )
        main_logger(self._evaluate_start_msg(tracker=tracker, **kwargs))

        ## Outputs:
        evaluated_predictions: Optional[Predictions] = None
        evaluated_metrics: Optional[List[Metric]] = None

        try:
            actors_were_created_in_this_call: bool = self.init_model(progress_bar=progress_bar, **kwargs)
            num_actors_created: int = len(self.model)

            if actors_were_created_in_this_call:
                main_logger(f"Created {num_actors_created} process actors.")

            dataset: Dataset = dataset.read(read_as=read_as, npartitions=num_actors_created)
            data: ScalableDataFrame = dataset.data
            input_len: int = len(data)
            input_len_str: str = String.readable_number(input_len, decimals=1, short=True)

            ## Submit data to be predicted:
            dataset_params: Dict = dataset.dict(exclude={"data"})

            predictions: List[Any] = self._run_evaluation_local(
                data=data,
                dataset_params=dataset_params,
                input_len=input_len,
                input_len_str=input_len_str,
                num_actors_created=num_actors_created,
                batch_size=batch_size,
                submission_batch_size=submission_batch_size,
                predictions_destination=predictions_destination,
                return_predictions=return_predictions,
                metrics=metrics,
                progress_bar=progress_bar,
                failure_action=failure_action,
                data_loading_strategy=DataLoadingStrategy.LOCAL,
                load_balancing_strategy=load_balancing_strategy,
                worker_queue_len=worker_queue_len,
                submission_batch_wait=submission_batch_wait,
                submission_batch_wait_jitter=submission_batch_wait_jitter,
                evaluation_timeout=evaluation_timeout,
                main_logger=main_logger,
                debug_logger=debug_logger,
                **kwargs,
            )

            if return_predictions or metrics is not None:
                debug_logger(f"Collecting {len(predictions)} predictions...")
                accumulate_progress_bar: ProgressBar = ProgressBar.of(
                    progress_bar,
                    total=input_len,
                    desc=f"Collecting {input_len_str} rows",
                    initial=0,
                )
                evaluated_predictions: List[Predictions] = []
                evaluated_metrics: List[Metric] = []

                for pred_i, pred in enumerate(predictions):
                    debug_logger(f"Collecting prediction#{pred_i}: type={type(pred)}")
                    try:
                        pred_result: Optional[Predictions] = get_result(pred, wait=60.0)
                    except Exception as e:
                        main_logger(
                            f"Error while collecting prediction#{pred_i}:\n{String.format_exception_msg(e)}"
                        )
                        raise e

                    if pred_result is None:
                        debug_logger(f"Collected prediction#{pred_i}: found None.")
                    else:
                        debug_logger(f"Collected prediction#{pred_i}: {type(pred_result)}.")
                        evaluated_predictions.append(pred_result)
                        accumulate_progress_bar.update(len(pred_result))

                if len(evaluated_predictions) == 0:
                    debug_logger(f"No results. evaluated_predictions={evaluated_predictions}")
                    accumulate_progress_bar.failed("No results")
                    raise ValueError("All predictions returned from actors were None.")

                evaluated_predictions: Predictions = Predictions.concat(
                    evaluated_predictions, error_on_empty=True
                )
                debug_logger(f"Concatenated into {len(evaluated_predictions)} rows of predictions.")

                if len(evaluated_predictions) != input_len:
                    num_failed_rows: int = input_len - len(evaluated_predictions)
                    num_failed_rows_str: str = String.readable_number(num_failed_rows, decimals=1, short=True)
                    accumulate_progress_bar.failed(f"Failed for {num_failed_rows_str} rows")
                    if allow_partial_predictions is False:
                        raise ValueError(
                            f"Partial predictions returned: expected {input_len} rows, "
                            f"but only got {len(evaluated_predictions)} rows from actors."
                        )
                else:
                    accumulate_progress_bar.success(f"Collected {input_len_str} rows")

                if metrics is not None:
                    for metric in metrics:
                        evaluated_metrics.append(evaluated_predictions.evaluate(metric=metric))
            else:
                ## Wait for predictions to complete:
                wait(predictions)

            timer.stop()
            main_logger(
                self._evaluate_end_msg(
                    input_len=input_len,
                    timer=timer,
                    num_actors_created=num_actors_created,
                    tracker=tracker,
                )
            )
            return evaluated_predictions, evaluated_metrics
        except Exception as e:
            error_msg: str = f"Error during evaluation:\n{String.format_exception_msg(e)}"
            main_logger(error_msg)
            raise e
        except KeyboardInterrupt as e:
            raise e
        finally:
            ## If we don't have a timeout, delete actors after every execution.
            if self.cache_timeout is None:
                self.cleanup_model()
            return evaluated_predictions, evaluated_metrics

    def _run_evaluation_local(
        self,
        *,
        data: ScalableDataFrame,
        dataset_params: Dict,
        input_len: int,
        input_len_str: str,
        num_actors_created: int,
        batch_size: int,
        submission_batch_size: int,
        predictions_destination: Optional[FileMetadata],
        return_predictions: bool,
        metrics: Optional[List[Metric]],
        progress_bar: Union[Dict, bool],
        failure_action: FailureAction,
        data_loading_strategy: DataLoadingStrategy,
        load_balancing_strategy: LoadBalancingStrategy,
        worker_queue_len: conint(ge=0),
        submission_batch_wait: confloat(ge=0),
        submission_batch_wait_jitter: confloat(ge=0.0, le=1.0),
        evaluation_timeout: confloat(ge=0, allow_inf_nan=True),
        main_logger: Callable,
        debug_logger: Callable,
        **kwargs,
    ) -> List[Any]:
        """
        Run evaluation using local data loading strategy with multiple processes.

        Args:
            data: Input data to process
            dataset_params: Parameters for creating datasets
            input_len: Total number of rows in the dataset
            input_len_str: Human-readable string of input length
            num_actors_created: Number of process actors created
            batch_size: Batch size for model inference
            submission_batch_size: Size of batches submitted to actors
            predictions_destination: Optional file destination for predictions
            return_predictions: Whether to return prediction results
            metrics: Optional metrics to evaluate
            progress_bar: Progress bar configuration
            failure_action: Action to take on failure
            data_loading_strategy: Strategy for data loading
            load_balancing_strategy: Strategy for load balancing across actors
            worker_queue_len: Maximum queue length per worker
            submission_batch_wait: Wait time when all actors are busy
            submission_batch_wait_jitter: Jitter factor for wait time
            evaluation_timeout: Maximum time to wait for evaluation
            main_logger: Logger for important messages
            debug_logger: Logger for debug messages
            **kwargs: Additional arguments passed to evaluate_shard

        Returns:
            List of futures containing prediction results
        """
        # print(f"Evaluating {len(data)} rows ({type(data)}) using {num_actors_created} processes")

        ## Consolidated tracking structure:
        ## - Map actor_idx -> {actor, futures_dict}
        ## - futures_dict maps batch_idx -> future
        ## This allows us to track both actor load and preserve result ordering
        futures_info: Dict[int, Dict[str, Union[Any, Dict[int, Any]]]] = {
            actor_i: {
                "actor": actor,
                "futures": {},  # batch_idx -> future mapping
                "batch_sizes": {},  # batch_idx -> number of rows in batch
            }
            for actor_i, actor in enumerate(self.model)
        }

        rows_submitted: int = 0
        rows_completed: int = 0
        submissions_progress_bar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=input_len,
            desc=f"Submitting {input_len_str} rows",
            unit="batch",
        )
        rows_completed_progress_bar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=input_len,
            desc=f"Evaluating {input_len_str} rows",
            initial=0,
        )
        ## Track results in submission order:
        predictions: List[Any] = []

        ## Load each shard of data and send to processes:
        batch_idx: int = 0
        for batch_data in data.stream(
            batch_size=submission_batch_size,
            shuffle=False,
            stream_as=DataLayout.PANDAS,
            fetch_partitions=1,
        ):
            # print(
            #     f">> Evaluating batch #{batch_idx} with {len(batch_data)} rows "
            #     f"with {load_balancing_strategy=}"
            # )
            ## Select actor based on load balancing strategy
            if load_balancing_strategy is LoadBalancingStrategy.ROUND_ROBIN:
                selected_actor_idx: int = batch_idx % num_actors_created
            elif load_balancing_strategy is LoadBalancingStrategy.RANDOM:
                selected_actor_idx: int = random.randrange(num_actors_created)
            elif load_balancing_strategy is LoadBalancingStrategy.LEAST_USED:
                ## Cleanup any completed futures before submitting a new batch
                rows_completed = self._cleanup_pending_futures(
                    futures_info,
                    rows_completed=rows_completed,
                    rows_completed_progress_bar=rows_completed_progress_bar,
                    debug_logger=debug_logger,
                    main_logger=main_logger,
                )
                ## Find actors with minimum workload:
                min_pending_futs, candidate_actor_idxs = self._min_pending_futures(
                    futures_info,
                    worker_queue_len=worker_queue_len,
                )
                ## If all actors are at capacity, wait for some to complete:
                while min_pending_futs >= worker_queue_len:
                    debug_logger(
                        f"All actors have {min_pending_futs}+ tasks (above worker_queue_len={worker_queue_len}), "
                        f"waiting for {submission_batch_wait} seconds."
                    )
                    ## Wait and recheck future completion status:
                    time_to_wait: float = np.random.uniform(
                        submission_batch_wait * (1 - submission_batch_wait_jitter),
                        submission_batch_wait * (1 + submission_batch_wait_jitter),
                    )
                    time.sleep(time_to_wait)
                    rows_completed = self._cleanup_pending_futures(
                        futures_info,
                        rows_completed=rows_completed,
                        rows_completed_progress_bar=rows_completed_progress_bar,
                        debug_logger=debug_logger,
                        main_logger=main_logger,
                    )
                    min_pending_futs, candidate_actor_idxs = self._min_pending_futures(
                        futures_info,
                        worker_queue_len=worker_queue_len,
                    )
                ## Choose randomly among least busy actors:
                selected_actor_idx: int = random.choice(candidate_actor_idxs)

                if self.verbosity >= 3:
                    pending_counts: Dict = {idx: len(info["futures"]) for idx, info in futures_info.items()}
                    debug_logger(
                        f"Actor workloads: {pending_counts}\n"
                        f">> Submitting batch#{batch_idx} ({len(batch_data)} rows, batch_size={batch_size}) "
                        f"to actor {selected_actor_idx}"
                    )
            else:
                raise NotImplementedError(f"Unsupported `load_balancing_strategy`: {load_balancing_strategy}")

            ## Get the actor and submit the task
            selected_actor: Any = futures_info[selected_actor_idx]["actor"]
            fut: Any = selected_actor.evaluate_shard.remote(
                batch_data,
                dataset_params={
                    **dataset_params,
                    **dict(data_idx=batch_idx),
                },
                input_len=input_len,
                batch_size=batch_size,
                predictions_destination=predictions_destination,
                return_predictions=return_predictions or metrics is not None,
                failure_action=failure_action,
                data_loading_strategy=data_loading_strategy,
                **kwargs,
            )

            ## Track the future with its batch index and size,
            ## both for load balancing and order preservation:
            futures_info[selected_actor_idx]["futures"][batch_idx] = fut
            futures_info[selected_actor_idx]["batch_sizes"][batch_idx] = len(batch_data)
            rows_submitted += len(batch_data)
            submissions_progress_bar.update(len(batch_data))
            ## Keep predictions list in submission order:
            predictions.append(fut)
            batch_idx += 1
        submissions_progress_bar.success(f"Submitted {input_len_str} rows")

        ## Check for completed futures periodically during submission:
        ## Track till all rows are completed:
        rows_completed_start_time: float = time.time()
        while (
            rows_completed < rows_submitted and time.time() < rows_completed_start_time + evaluation_timeout
        ):
            rows_completed = self._cleanup_pending_futures(
                futures_info,
                rows_completed=rows_completed,
                rows_completed_progress_bar=rows_completed_progress_bar,
                debug_logger=debug_logger,
                main_logger=main_logger,
            )
            time.sleep(self.progress_update_frequency)
        rows_completed_progress_bar.success(f"Evaluated {input_len_str} rows")
        return predictions

    @classmethod
    def _cleanup_pending_futures(
        cls,
        futures_info: Dict,
        *,
        rows_completed: int,
        rows_completed_progress_bar: ProgressBar,
        debug_logger: Callable,
        main_logger: Callable,
    ) -> int:  # Returns updated rows_completed
        ## Collect all futures along with their actor_idx and batch_idx
        futures_to_check = {}
        for actor_idx in futures_info.keys():
            for batch_idx, fut in futures_info[actor_idx]["futures"].items():
                futures_to_check[(actor_idx, batch_idx)] = fut

        if len(futures_to_check) == 0:
            return rows_completed

        ## Check all futures for completion
        for (actor_idx, batch_idx), fut in futures_to_check.items():
            if is_done(fut):
                try:
                    ## Fetch the result; will throw an exception if task failed:
                    result = get_result(fut)
                except Exception as e:
                    main_logger(f"Error in batch #{batch_idx}: {String.format_exception_msg(e)}")
                finally:
                    ## Remove the future from tracking since it's completed
                    batch_size = futures_info[actor_idx]["batch_sizes"].pop(batch_idx, 0)
                    futures_info[actor_idx]["futures"].pop(batch_idx, None)
                    rows_completed += batch_size
                    rows_completed_progress_bar.update(batch_size)

        return rows_completed

    @classmethod
    def _min_pending_futures(
        cls,
        futures_info: Dict,
        *,
        worker_queue_len: int,
    ) -> Tuple[int, List[int]]:
        ## Find actors with the minimum pending number of futures
        candidate_actor_idxs: List[int] = []
        min_pending_futs: int = worker_queue_len + 1
        for actor_idx, info in futures_info.items():
            pending_count = len(info["futures"])
            if pending_count < min_pending_futs:
                min_pending_futs = pending_count
                candidate_actor_idxs = [actor_idx]
            elif pending_count == min_pending_futs:
                candidate_actor_idxs.append(actor_idx)
        return min_pending_futs, candidate_actor_idxs

    def _run_evaluation_progress_bar(self, progress_bar: Optional[Dict], **kwargs) -> Union[Dict, bool]:
        if self.verbosity >= 2:
            return progress_bar
        return False

    def _evaluate_start_msg(self, *, tracker: Tracker, **kwargs) -> str:
        if tracker.tracker_name == "noop":
            tracker_msg: str = "Logs will not be tracked."
        else:
            tracker_msg: str = f'{tracker.class_name}@{tracker.id} will save logs to: "{tracker.log_dir}"'
        return (
            f"\nEvaluating using nested evaluator: "
            f"{String.pretty(self._create_nested_evaluator_params(**kwargs))}"
            f"\n{tracker_msg}"
        )

    def _evaluate_end_msg(
        self,
        *,
        input_len: int,
        timer: Timer,
        num_actors_created: int,
        tracker: Tracker,
        **kwargs,
    ) -> str:
        if tracker.tracker_name == "noop":
            tracker_msg: str = "Logs have not been tracked."
        else:
            tracker_msg: str = f'{tracker.class_name}@{tracker.id} has saved logs to "{tracker.log_dir}"'
        return (
            f"Evaluated {input_len} rows in {timer.time_taken_str} "
            f"using {num_actors_created} processes "
            f"({input_len / timer.time_taken_sec:.3f} rows/sec or "
            f"{input_len / (num_actors_created * timer.time_taken_sec):.3f} rows/sec/process)\n{tracker_msg}"
        )
