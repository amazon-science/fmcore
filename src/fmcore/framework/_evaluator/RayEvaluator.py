import gc
import math
import random
import time
import warnings
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from bears import FileMetadata
from bears.core.frame import DaskScalableDataFrame, ScalableDataFrame
from bears.util import (
    DataLoadingStrategy,
    LoadBalancingStrategy,
    ProgressBar,
    RayActorComposite,
    RayInitConfig,
    RayResources,
    RequestCounter,
    ShardingStrategy,
    String,
    Timeout,
    Timer,
    accumulate,
    get_default,
    get_result,
    max_num_resource_actors,
    safe_validate_arguments,
    set_param_from_alias,
    wait,
)
from bears.util.language._import import _IS_RAY_INSTALLED
from pydantic import ConfigDict, confloat, conint, model_validator

from fmcore.constants import (
    REMOTE_STORAGES,
    DataLayout,
    FailureAction,
    Storage,
)
from fmcore.framework._dataset import Dataset
from fmcore.framework._evaluator.Evaluator import Evaluator
from fmcore.framework._metric import Metric
from fmcore.framework._predictions import Predictions
from fmcore.framework._tracker.Tracker import Tracker

from .ParallelEvaluator import ParallelAlgorithmEvaluator

RayEvaluator = "RayEvaluator"
if _IS_RAY_INSTALLED:
    import ray

    @ray.remote
    class RowCounter:
        def __init__(self):
            self.rows_completed: int = 0

        def increment_rows(self, num_rows: int):
            self.rows_completed += num_rows
            # print(f'Shard {shard[0]} slept for {time_slept:.3f} sec and completed {num_rows} rows')

        def get_rows_completed(self) -> int:
            return self.rows_completed

    @ray.remote
    class RayAlgorithmEvaluator(ParallelAlgorithmEvaluator):
        """
        A Ray-based algorithm evaluator that handles distributed evaluation of model predictions.

        Example usage:
            >>> evaluator_params = {"evaluator": "local", "task": "generation", "AlgorithmClass": "HFGenerativeLM"}
            >>> request_counter = RequestCounter.remote()
            >>> actor = RayAlgorithmEvaluator.remote(evaluator=evaluator_params, actor=(0, 1),
            ...                                     request_counter=request_counter, verbosity=2)
            >>> future = actor.evaluate_shard.remote(data, dataset_params=dataset_params, batch_size=4)
        """

        def __init__(self, *, request_counter: RequestCounter, **kwargs):
            self.request_counter: RequestCounter = request_counter
            super().__init__(**kwargs)

        def get_ip_address(self) -> Optional[str]:
            try:
                ## Ref: https://discuss.ray.io/t/get-current-executor-ip/4916
                return ray.util.get_node_ip_address()
            except Exception:
                return None

        def _update_row_counter(self, row_counter: RowCounter, num_rows: int) -> None:
            row_counter.increment_rows.remote(num_rows=num_rows)

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
            row_counter: Optional[RowCounter],
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
                row_counter: Ray actor for counting processed rows
                failure_action: Action to take on failure
                data_loading_strategy: Strategy for loading data
                **kwargs: Additional arguments passed to the evaluator

            Returns:
                Predictions or None depending on return_predictions
            """
            self.request_counter.started_request.remote()

            data_loading_strategy = DataLoadingStrategy(data_loading_strategy)
            is_sharded = data_loading_strategy is DataLoadingStrategy.DASK
            shard = self.actor if is_sharded else (0, 1)

            data: ScalableDataFrame = ScalableDataFrame.of(data)
            try:
                return self._evaluate_batch_stream(
                    data=data,
                    dataset_params=dataset_params,
                    input_len=input_len,
                    batch_size=batch_size,
                    batches_per_save=batches_per_save,
                    predictions_destination=predictions_destination,
                    return_predictions=return_predictions,
                    failure_action=failure_action,
                    is_sharded=is_sharded,
                    shard=shard,
                    row_counter=row_counter,
                    **kwargs,
                )
            finally:
                self.request_counter.completed_request.remote()

    class RayEvaluator(Evaluator):
        aliases = ["ray"]

        model_config = ConfigDict(
            extra="allow",
        )

        class RunConfig(Evaluator.RunConfig):
            ray_init: RayInitConfig = {}

        nested_evaluator_name: Optional[str] = None
        num_models: Optional[conint(ge=1)] = None
        model: Optional[List[RayActorComposite]] = None  ## Stores the actors.
        resources_per_model: RayResources = RayResources(num_cpus=1, num_gpus=0)
        progress_update_frequency: confloat(ge=0.0) = 15.0
        ## By default, do not cache the model:
        cache_timeout: Optional[Union[Timeout, confloat(gt=0)]] = None

        @model_validator(mode="before")
        @classmethod
        def ray_evaluator_params(cls, params: Dict) -> Dict:
            params: Dict = cls._set_common_evaluator_params(params)
            set_param_from_alias(params, param="nested_evaluator_name", alias=["nested_evaluator"])
            set_param_from_alias(
                params,
                param="num_models",
                alias=[
                    "max_models",
                    "max_num_models",
                    "model_copies",
                    "num_copies",
                    "num_actors",
                ],
            )
            set_param_from_alias(
                params,
                param="resources_per_model",
                alias=[
                    "model_resources",
                    "resources",
                ],
            )
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
            if params.get("device") is not None:
                raise ValueError(
                    f'Do not pass "device" to {cls.class_name}.of(), '
                    f"instead pass it as: {cls.class_name}.evaluate(device=...)"
                )

            return params

        def initialize(self, **kwargs):
            if self.model_num_gpus <= 1 or self.AlgorithmClass.class_name == "VLLMGenerativeLM":
                self.nested_evaluator_name: str = get_default(self.nested_evaluator_name, "local")
            else:
                self.nested_evaluator_name: str = get_default(self.nested_evaluator_name, "accelerate")
            if not ray.is_initialized():
                raise SystemError("ray cluster is not initialized.")

        def _load_model(
            self,
            *,
            num_actors: Optional[int] = None,
            progress_bar: Optional[Union[Dict, bool]] = True,
            **kwargs,
        ) -> List[RayActorComposite]:
            num_actors: int = get_default(num_actors, self.num_actors)
            progress_bar: Union[Dict, bool] = self._run_evaluation_progress_bar(progress_bar)
            nested_evaluator_params: Dict = self._create_nested_evaluator_params(**kwargs)

            def actor_factory(*, request_counter: Any, actor_i: int, actor_id: str, **kwargs):
                return RayAlgorithmEvaluator.options(
                    num_cpus=self.model_num_cpus,
                    num_gpus=self.model_num_gpus,
                ).remote(
                    evaluator=nested_evaluator_params,
                    actor=(actor_i, num_actors),
                    request_counter=request_counter,
                    verbosity=self.verbosity,
                )

            return RayActorComposite.create_actors(
                actor_factory,
                num_actors=num_actors,
                progress_bar=progress_bar,
            )

        def cleanup_model(
            self,
        ):
            self._kill_actors()

        def _kill_actors(self):
            try:
                if self.model is not None:
                    actor_composites: List[RayActorComposite] = self.model
                    self.model: Optional[List[RayActorComposite]] = None
                    for actor_comp in actor_composites:
                        actor_comp.kill()
                        del actor_comp
                    del actor_composites
            finally:
                gc.collect()

        @property
        def max_num_actors(self) -> int:
            cluster_resources: Dict = ray.cluster_resources()
            ray_cluster_num_cpus: int = int(cluster_resources["CPU"])
            ray_cluster_num_gpus: int = int(cluster_resources.get("GPU", 0))

            max_num_cpu_actors: int = max_num_resource_actors(
                self.resources_per_model.num_cpus,
                ray_cluster_num_cpus,
            )
            max_num_gpu_actors: Union[int, float] = max_num_resource_actors(
                self.resources_per_model.num_gpus,
                ray_cluster_num_gpus,
            )
            max_num_actors: int = min(max_num_gpu_actors, max_num_cpu_actors)
            return max_num_actors

        @property
        def model_num_cpus(self) -> Union[conint(ge=1), confloat(ge=0.0, lt=1.0)]:
            return self.resources_per_model.dict().get("cpu", 1)

        @property
        def model_num_gpus(self) -> Union[conint(ge=0), confloat(ge=0.0, lt=1.0)]:
            return self.resources_per_model.num_gpus

        @property
        def num_actors(self) -> int:
            cluster_resources: Dict = ray.cluster_resources()
            ray_cluster_num_cpus: int = int(cluster_resources["CPU"])
            ray_cluster_num_gpus: int = int(cluster_resources.get("GPU", 0))

            model_num_cpus: Union[conint(ge=1), confloat(ge=0.0, lt=1.0)] = self.model_num_cpus
            model_num_gpus: Union[conint(ge=0), confloat(ge=0.0, lt=1.0)] = self.model_num_gpus
            max_num_actors: int = self.max_num_actors
            num_actors: Optional[int] = self.num_models
            if num_actors is None:
                warnings.warn(
                    f"`num_models` is not specified. Since each model-copy requires "
                    f"{model_num_cpus} cpus and {model_num_gpus} gpus, we create {max_num_actors} model-copies so as "
                    f"to utilize the entire Ray cluster (having {ray_cluster_num_cpus} cpus and {ray_cluster_num_gpus} gpus). "
                    f"To reduce the cluster-utilization, explicitly pass `num_models`."
                )
                num_actors: int = max_num_actors
            elif num_actors > max_num_actors:
                warnings.warn(
                    f"Requested {num_actors} model-copies (each with {model_num_cpus} cpus and {model_num_gpus} gpus); "
                    f"however, the Ray cluster only has {ray_cluster_num_cpus} cpus and {ray_cluster_num_gpus} gpus, "
                    f"thus we can create at most {max_num_actors} model-copies."
                )
            num_actors: int = min(num_actors, max_num_actors)
            return num_actors

        def _create_nested_evaluator_params(self, **kwargs) -> Dict:
            nested_evaluator_name: str = self.nested_evaluator_name
            if self.model_dir is not None and not self.model_dir.is_remote_storage():
                raise ValueError(
                    f"When passing `model_dir` to {self.class_name}.of(...), the model directory "
                    f"must be on a remote storage, i.e. one of: {REMOTE_STORAGES}"
                )
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
            # print(f'nested_evaluator_params dict:\n{nested_evaluator_params}')
            nested_evaluator_params["evaluator"]: str = nested_evaluator_name
            if self.model_num_gpus > 0:
                nested_evaluator_params.setdefault("device", "cuda")
            return nested_evaluator_params

        @staticmethod
        def ray_logger(text: str, should_log: bool, tracker: Tracker):
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
            sharding_strategy: ShardingStrategy = ShardingStrategy.COARSE,
            data_loading_strategy: DataLoadingStrategy = DataLoadingStrategy.LOCAL,
            load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_USED,
            read_as: Optional[DataLayout] = DataLayout.PANDAS,
            submission_batch_size: Optional[conint(ge=1)] = None,
            worker_queue_len: conint(ge=0) = 2,
            submission_batch_wait: confloat(ge=0) = 15,
            submission_batch_wait_jitter: confloat(ge=0.0, le=1.0) = 0.05,
            evaluation_timeout: confloat(ge=0, allow_inf_nan=True) = math.inf,
            allow_partial_predictions: bool = False,
            **kwargs,
        ) -> Tuple[Optional[Predictions], Optional[List[Metric]]]:
            ## TODO: add rows per save, SaveStrategy and UpdateStrategy:
            set_param_from_alias(kwargs, param="batches_per_save", alias=["batches_per_update"], default=1)
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
                self.ray_logger,
                ## Unless we request silence (verbosity=0), print important information.
                should_log=self.verbosity >= 2,
                tracker=tracker,
            )
            ## Verbosity >= 3: detailed logging
            debug_logger: Callable = partial(
                self.ray_logger,
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
                    resource_req_str: str = String.join_human(
                        [
                            f"{resource_req} {resource_name}(s)"
                            for resource_name, resource_req in self.resources_per_model.dict().items()
                        ]
                    )
                    main_logger(
                        f"Created {num_actors_created} copies of the model, each using {resource_req_str}."
                    )
                dataset: Dataset = dataset.read(read_as=read_as, npartitions=num_actors_created)
                data: ScalableDataFrame = dataset.data.persist(wait=True)
                input_len: int = len(data)
                input_len_str: str = String.readable_number(input_len, decimals=1, short=True)
                if sharding_strategy is ShardingStrategy.GRANULAR:
                    if not isinstance(data, DaskScalableDataFrame):
                        raise ValueError(
                            f"Can only use sharding_strategy={ShardingStrategy.GRANULAR} when read_as={DataLayout.DASK}; "
                            f"found read_as={read_as}."
                        )
                    sharding_timer: Timer = Timer(silent=True)
                    sharding_timer.start()
                    _, _ = data.set_shard_divisions(
                        num_shards=num_actors_created,
                        num_rows=batch_size,
                        inplace=True,
                    )
                    sharding_timer.stop()
                    debug_logger(f"Set shard divisions in {sharding_timer.time_taken_human}.")

                ## Submit data to be predicted:
                if self.model_num_gpus > 0:
                    kwargs.setdefault("device", "cuda")
                dataset_params: Dict = dataset.dict(exclude={"data"})

                if data_loading_strategy is DataLoadingStrategy.DASK:
                    predictions: List[ray.ObjectRef] = self._run_evaluation_dask(
                        data=data,
                        dataset_params=dataset_params,
                        input_len=input_len,
                        input_len_str=input_len_str,
                        num_actors_created=num_actors_created,
                        batch_size=batch_size,
                        predictions_destination=predictions_destination,
                        return_predictions=return_predictions,
                        metrics=metrics,
                        progress_bar=progress_bar,
                        failure_action=failure_action,
                        data_loading_strategy=data_loading_strategy,
                        worker_queue_len=worker_queue_len,
                        evaluation_timeout=evaluation_timeout,
                        main_logger=main_logger,
                        debug_logger=debug_logger,
                        **kwargs,
                    )
                elif data_loading_strategy is DataLoadingStrategy.LOCAL:
                    predictions: List[ray.ObjectRef] = self._run_evaluation_local(
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
                        data_loading_strategy=data_loading_strategy,
                        load_balancing_strategy=load_balancing_strategy,
                        worker_queue_len=worker_queue_len,
                        submission_batch_wait=submission_batch_wait,
                        submission_batch_wait_jitter=submission_batch_wait_jitter,
                        evaluation_timeout=evaluation_timeout,
                        main_logger=main_logger,
                        debug_logger=debug_logger,
                        **kwargs,
                    )
                else:
                    raise NotImplementedError(f"Unsupported `data_loading_strategy`: {data_loading_strategy}")

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
                        debug_logger(f"Collecting prediction#{pred_i}: type={type(pred)}, val={pred}")
                        try:
                            pred: Optional[Predictions] = get_result(pred, wait=60.0)
                        except Exception as e:
                            main_logger(
                                f"Error while collecting prediction#{pred_i}:\n{String.format_exception_msg(e)}"
                            )
                            raise e
                        if pred is None:
                            debug_logger(f"Collected prediction#{pred_i}: found None.")
                        else:
                            debug_logger(f"Collected prediction#{pred_i}: {type(pred)}.")
                            evaluated_predictions.append(pred)
                            accumulate_progress_bar.update(len(pred))
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
                        num_failed_rows_str: str = String.readable_number(
                            num_failed_rows, decimals=1, short=True
                        )
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
                raise e
            except KeyboardInterrupt as e:
                raise e
            finally:
                ## If we don't have a timeout, delete actors after every execution.
                if self.cache_timeout is None:
                    self.cleanup_model()
                return evaluated_predictions, evaluated_metrics

        def _run_evaluation_dask(
            self,
            *,
            data: ScalableDataFrame,
            dataset_params: Dict,
            read_as: DataLayout,
            input_len: int,
            input_len_str: str,
            num_actors_created: int,
            batch_size: int,
            predictions_destination: Optional[FileMetadata],
            return_predictions: bool,
            metrics: Optional[List[Metric]],
            progress_bar: Union[Dict, bool],
            failure_action: FailureAction,
            data_loading_strategy: DataLoadingStrategy,
            worker_queue_len: int,
            evaluation_timeout: float,
            main_logger: Callable,
            debug_logger: Callable,
            **kwargs,
        ) -> List[ray.ObjectRef]:
            try:
                row_counter: ray.actor.ActorHandle = RowCounter.options(
                    num_cpus=0.1,
                    max_concurrency=max(
                        num_actors_created + 2,
                        worker_queue_len * num_actors_created + 2,
                    ),
                ).remote()
                submissions_progress_bar: ProgressBar = ProgressBar.of(
                    progress_bar,
                    total=num_actors_created,
                    desc=f"Submitting {input_len_str} rows",
                    unit="batch",
                )
                ## Each actor streams data from Dask dataframe on the cluster:
                if not isinstance(data, DaskScalableDataFrame):
                    raise ValueError(
                        f"Can only use data_loading_strategy={DataLoadingStrategy.DASK} when read_as={DataLayout.DASK}; "
                        f"found read_as={read_as}."
                    )
                rows_completed: int = 0
                rows_completed_progress_bar: Optional[ProgressBar] = None
                predictions: List[ray.ObjectRef] = []
                ## When using DataLoadingStrategy.DASK, each actor will evaluate a fixed set of set, so
                ## LoadBalancingStrategy does not come into play.
                for actor_i, actor_comp in enumerate(self.model):
                    predictions.append(
                        actor_comp.actor.evaluate_shard.remote(
                            data,
                            dataset_params=dataset_params,
                            input_len=input_len,
                            batch_size=batch_size,
                            predictions_destination=predictions_destination,
                            return_predictions=return_predictions or metrics is not None,
                            row_counter=row_counter,
                            failure_action=failure_action,
                            data_loading_strategy=data_loading_strategy,
                            **kwargs,
                        )
                    )
                    submissions_progress_bar.update(1)
                ## Initialize with number of rows completed so far:
                rows_completed: int = get_result(row_counter.get_rows_completed.remote())
                rows_completed_progress_bar: ProgressBar = ProgressBar.of(
                    progress_bar,
                    total=input_len,
                    desc=f"Evaluating {input_len_str} rows",
                    initial=rows_completed,
                )
                submissions_progress_bar.success(f"Submitted {input_len_str} rows")
                ## Track till all rows are completed:
                rows_completed_start_time: float = time.time()
                while (
                    rows_completed < input_len
                    and time.time() < rows_completed_start_time + evaluation_timeout
                ):
                    time.sleep(self.progress_update_frequency)
                    new_rows_completed: int = get_result(row_counter.get_rows_completed.remote())
                    rows_completed_progress_bar.update(new_rows_completed - rows_completed)
                    rows_completed: int = new_rows_completed
                rows_completed_progress_bar.success(f"Evaluated {input_len_str} rows")
                return predictions
            finally:
                if "row_counter" in locals():
                    accumulate(ray.kill(row_counter))
                    del row_counter

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
            worker_queue_len: int,
            submission_batch_wait: float,
            submission_batch_wait_jitter: float,
            evaluation_timeout: float,
            main_logger: Callable,
            debug_logger: Callable,
            **kwargs,
        ) -> List[ray.ObjectRef]:
            ## Consolidated tracking structure:
            ## - Map actor_id -> {composite, futures_dict}
            ## - futures_dict maps batch_idx -> future
            ## This allows us to track both actor load and preserve result ordering
            futures_info: Dict[str, Dict[str, Union[RayActorComposite, Dict[int, Any]]]] = {
                actor_comp.actor_id: {
                    "composite": actor_comp,
                    "futures": {},  # batch_idx -> future mapping
                    "batch_sizes": {},  # batch_idx -> number of rows in batch
                }
                for actor_comp in self.model
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
            predictions: List[ray.ObjectRef] = []
            ## Load each shard of data on the calling machine, and send to the cluster:
            for batch_idx, batch_data in enumerate(
                data.stream(
                    batch_size=submission_batch_size,
                    shuffle=False,
                    stream_as=DataLayout.PANDAS,
                    fetch_partitions=1,
                )
            ):
                ## Select actor based on load balancing strategy
                if load_balancing_strategy is LoadBalancingStrategy.ROUND_ROBIN:
                    selected_actor_id: str = self.model[batch_idx % num_actors_created].actor_id
                elif load_balancing_strategy is LoadBalancingStrategy.RANDOM:
                    selected_actor_id: str = self.model[
                        random.choice(list(range(0, num_actors_created)))
                    ].actor_id
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
                    min_pending_futs, candidate_actor_ids = self._min_pending_futures(
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
                        min_pending_futs, candidate_actor_ids = self._min_pending_futures(
                            futures_info,
                            worker_queue_len=worker_queue_len,
                        )
                    ## Choose randomly among least busy actors:
                    selected_actor_id: str = random.choice(candidate_actor_ids)

                    if self.verbosity >= 3:
                        pending_counts: Dict = {
                            actor_id: len(info["futures"]) for actor_id, info in futures_info.items()
                        }
                        debug_logger(
                            f"Actor workloads: {pending_counts}\n"
                            f">> Submitting batch#{batch_idx} ({len(batch_data)} rows, batch_size={batch_size}) "
                            f"to actor '{selected_actor_id}' at IP address "
                            f"{get_result(futures_info[selected_actor_id]['composite'].actor.get_ip_address.remote())}"
                        )
                else:
                    raise NotImplementedError(
                        f"Unsupported `load_balancing_strategy`: {load_balancing_strategy}"
                    )

                ## Get the actor composite and submit the task
                selected_actor_comp: RayActorComposite = futures_info[selected_actor_id]["composite"]
                fut: Any = selected_actor_comp.actor.evaluate_shard.remote(
                    batch_data,
                    dataset_params={
                        **dataset_params,
                        **dict(data_idx=batch_idx),
                    },
                    input_len=input_len,
                    batch_size=batch_size,
                    predictions_destination=predictions_destination,
                    return_predictions=return_predictions or metrics is not None,
                    row_counter=None,
                    failure_action=failure_action,
                    data_loading_strategy=data_loading_strategy,
                    **kwargs,
                )

                ## Track the future with its batch index and size,
                ## both for load balancing and order preservation:
                futures_info[selected_actor_id]["futures"][batch_idx] = fut
                futures_info[selected_actor_id]["batch_sizes"][batch_idx] = len(batch_data)
                rows_submitted += len(batch_data)
                submissions_progress_bar.update(len(batch_data))
                ## Keep predictions list in submission order:
                predictions.append(fut)
            submissions_progress_bar.success(f"Submitted {input_len_str} rows")

            ## Check for completed futures periodically during submission:
            ## Track till all rows are completed:
            rows_completed_start_time: float = time.time()
            while (
                rows_completed < rows_submitted
                and time.time() < rows_completed_start_time + evaluation_timeout
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
        ) -> int:  # Changed from NoReturn to int since we're returning rows_completed
            ## Collect all futures along with their actor_id and batch_idx
            futures_info_flat: List[Tuple[str, int, Any, int]] = []
            for actor_id in futures_info.keys():
                for batch_idx in futures_info[actor_id]["futures"].keys():
                    futures_info_flat.append(
                        (
                            actor_id,
                            batch_idx,
                            futures_info[actor_id]["futures"][batch_idx],
                            futures_info[actor_id]["batch_sizes"][batch_idx],
                        )
                    )

            if len(futures_info_flat) == 0:
                return rows_completed  # Return the unchanged rows_completed

            ## Use ray.wait to efficiently check which futures are done;
            ## timeout=0 means it returns immediately with currently completed futures:
            completed_futures, _ = ray.wait(
                [item[2] for item in futures_info_flat],
                num_returns=len(futures_info_flat),
                timeout=0,
            )
            ## None of the pending futures have completed yet:
            if len(completed_futures) == 0:
                return rows_completed  # Return the unchanged rows_completed

            ## Check which futures completed successfully and update rows_completed
            for completed_fut in completed_futures:
                for actor_id, batch_idx, fut, batch_size in futures_info_flat:
                    if fut == completed_fut:
                        try:
                            ## Fetch the result; will throw an exception if task failed:
                            result = ray.get(completed_fut)
                        except Exception as e:
                            main_logger(f"Error in batch #{batch_idx}: {String.format_exception_msg(e)}")
                        finally:
                            ## Remove the future from tracking since it's completed (regardless of whether
                            ## it succeeded or failed):
                            rows_completed += batch_size
                            rows_completed_progress_bar.update(batch_size)
                            futures_info[actor_id]["futures"].pop(batch_idx)
                            futures_info[actor_id]["batch_sizes"].pop(batch_idx)
                        break
            return rows_completed

        @classmethod
        def _min_pending_futures(
            cls,
            futures_info: Dict,
            *,
            worker_queue_len: int,
        ) -> Tuple[int, List[str]]:
            ## Find actors with the minimum pending number of futures
            candidate_actor_ids: List[str] = []
            min_pending_futs: int = worker_queue_len + 1
            for actor_id, info in futures_info.items():
                pending_count = len(info["futures"])
                if pending_count < min_pending_futs:
                    min_pending_futs = pending_count
                    candidate_actor_ids = [actor_id]
                elif pending_count == min_pending_futs:
                    candidate_actor_ids.append(actor_id)
            return min_pending_futs, candidate_actor_ids

        def _get_actor_usages(self) -> List[Tuple[int, float, str]]:
            actor_usages: List[Tuple[int, float, str]] = accumulate(
                [
                    (
                        actor_comp.request_counter.num_pending_requests.remote(),
                        actor_comp.request_counter.last_completed_timestamp.remote(),
                        actor_comp.actor_id,
                    )
                    for actor_comp in self.model
                ]
            )
            return actor_usages

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
                f"using {num_actors_created} model-copies "
                f"({input_len / timer.time_taken_sec:.3f} rows/sec or "
                f"{input_len / (num_actors_created * timer.time_taken_sec):.3f} rows/sec/copy)\n{tracker_msg}"
            )
