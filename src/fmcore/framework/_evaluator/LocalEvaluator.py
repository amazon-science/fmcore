import gc
import os
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

from bears import FileMetadata
from bears.util import Timeout, Timeout24Hr, Timer, confloat, get_default, safe_validate_arguments
from pydantic import model_validator

from fmcore.framework._algorithm import Algorithm
from fmcore.framework._evaluator.Evaluator import Evaluator
from fmcore.framework._metric import Metric
from fmcore.framework._predictions import Predictions
from fmcore.framework._tracker.Tracker import Tracker


class LocalEvaluator(Evaluator):
    aliases = ["local", "SimpleEvaluator", "simple"]

    ## Cache model locally for 15 mins:
    cache_timeout: Optional[Union[Timeout, confloat(gt=0)]] = Timeout24Hr(timeout=3 * 60 * 60)

    @model_validator(mode="before")
    @classmethod
    def set_local_evaluator_params(cls, params: Dict) -> Dict:
        params: Dict = cls._set_common_evaluator_params(params)
        return params

    def _load_model(
        self,
        *,
        cache_dir: Optional[Union[FileMetadata, Dict, str]] = None,
        **kwargs,
    ) -> Algorithm:
        kwargs.pop("model_dir", None)
        cache_dir: FileMetadata = FileMetadata.of(get_default(cache_dir, self.cache_dir)).mkdir(
            return_metadata=True
        )
        return Algorithm.of(
            **{
                **dict(
                    task=self.task,
                    algorithm=self.AlgorithmClass,
                    hyperparams=self.hyperparams,
                    model_dir=self.download_remote_model_to_cache_dir(**kwargs),
                    cache_dir=cache_dir,
                ),
                **kwargs,
            }
        )

    @staticmethod
    def local_logger(text: str, verbosity: int, tracker: Tracker):
        pid: int = os.getpid()
        text: str = f"(pid={pid}): {text}"
        if verbosity == 0:  ## Don't log anything.
            return
        else:
            tracker.info(text)

    @safe_validate_arguments
    def _run_evaluation(
        self,
        dataset: Any,
        *,
        tracker: Tracker,
        metrics: Optional[List[Metric]],
        return_predictions: bool,
        predictions_destination: Optional[FileMetadata],
        progress_bar: Optional[Dict],
        **kwargs,
    ) -> Tuple[Optional[Predictions], Optional[List[Metric]]]:
        timer: Timer = Timer(silent=True)
        timer.start()
        self.init_model(**kwargs)
        model: Algorithm = self.model
        logger: Callable = partial(
            self.local_logger,
            ## Unless we request silence (verbosity=0), print important information.
            verbosity=0 if self.verbosity == 0 else 1,
            tracker=tracker,
        )
        batch_logger: Callable = partial(
            self.local_logger,
            ## Skip batch-level logging unless we explicitly ask for output from all batches (verbosity=2).
            verbosity=2 if self.verbosity == 2 else 0,
            tracker=tracker,
        )
        ## Show progress bar only when printing important information.
        logger(self._evaluate_start_msg(model=model, tracker=tracker))
        progress_bar: Optional[Dict] = self._local_evaluator_progress_bar(progress_bar)
        try:
            ## Call chain: ._evaluate_single_model() -> Algorithm.predict() -> Algorithm.predict_iter()
            evaluated_predictions, evaluated_metrics, evaluated_num_rows = self._evaluate_single_model(
                model=model,
                dataset=dataset,
                metrics=metrics,
                return_predictions=return_predictions,
                predictions_destination=predictions_destination,
                logger=logger,
                batch_logger=batch_logger,
                progress_bar=progress_bar,
                **kwargs,
            )
            timer.stop()
            logger(
                self._evaluate_end_msg(
                    model=model,
                    timer=timer,
                    evaluated_num_rows=evaluated_num_rows,
                    tracker=tracker,
                )
            )
            return evaluated_predictions, evaluated_metrics
        except KeyboardInterrupt as e:
            raise e
        finally:
            if self.cache_timeout is None:  ## If we don't have a timeout, delete model after every execution.
                self.cleanup_model()

    def _local_evaluator_progress_bar(
        self,
        progress_bar: Optional[Dict],
        **kwargs,
    ) -> Optional[Dict]:
        if self.verbosity == 1 and progress_bar is not None:
            if not isinstance(progress_bar, dict):
                progress_bar: Optional[Dict] = dict()
        else:
            progress_bar: Optional[Dict] = None
        return progress_bar

    def cleanup_model(self):
        try:
            model: Optional[Algorithm] = self.model
            self.model = None
            if model is not None:
                model.cleanup()
                del model
        finally:
            gc.collect()

    def _evaluate_start_msg(self, *, model: Algorithm, tracker: Tracker, **kwargs) -> str:
        if tracker.tracker_name == "noop":
            tracker_msg: str = "Logs will not be tracked."
        else:
            tracker_msg: str = f'{tracker.class_name}@{tracker.id} will save logs to: "{tracker.log_dir}"'
        return f"\nEvaluating {model.task} model...\n{str(model)}\n{tracker_msg}"

    def _evaluate_end_msg(
        self,
        *,
        model: Algorithm,
        timer: Timer,
        evaluated_num_rows: int,
        tracker: Tracker,
        **kwargs,
    ) -> str:
        if tracker.tracker_name == "noop":
            tracker_msg: str = "Logs have not been tracked."
        else:
            tracker_msg: str = f'{tracker.class_name}@{tracker.id} has saved logs to "{tracker.log_dir}"'
        return (
            f"...model evaluation on {evaluated_num_rows} rows completed in {timer.time_taken_str} "
            f"({evaluated_num_rows / timer.time_taken_sec:.3f} rows/sec, including overhead time).\n"
            f"{tracker_msg}"
        )
