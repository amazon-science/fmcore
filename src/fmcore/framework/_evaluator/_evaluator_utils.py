import gc
import logging
import math
import random
import time
from contextlib import ExitStack, contextmanager
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
from autoenum import AutoEnum, auto
from bears import FileMetadata
from bears.core.frame import ScalableDataFrame
from bears.util import (
    LoadBalancingStrategy,
    ProgressBar,
    String,
    Timeout,
    Timer,
    accumulate,
    as_list,
    get_default,
    get_result,
    ignore_all_output,
    ignore_logging,
    ignore_stdout,
    ignore_warnings,
    safe_validate_arguments,
    set_param_from_alias,
    wait,
)
from bears.util.aws import S3Util
from bears.util.concurrency._processes import actor
from pydantic import ConfigDict, confloat, conint, model_validator

from fmcore.constants import (
    FILE_FORMAT_TO_FILE_ENDING_MAP,
    DataLayout,
    FailureAction,
    Storage,
)
from fmcore.framework._dataset import Dataset
from fmcore.framework._evaluator.Evaluator import Evaluator, save_predictions
from fmcore.framework._metric import Metric
from fmcore.framework._predictions import Predictions, load_predictions
from fmcore.framework._tracker.Tracker import Tracker


ALGORITHM_EVALUATOR_VERBOSITY_IGNORE: Dict[int, List[Callable]] = {
    0: [ignore_all_output],
    1: [ignore_stdout, ignore_warnings, partial(ignore_logging, disable_upto=logging.WARNING)],
    2: [partial(ignore_logging, disable_upto=logging.DEBUG)],
    3: [partial(ignore_logging, disable_upto=logging.NOTSET)],
    4: [partial(ignore_logging, disable_upto=logging.NOTSET)],
    5: [partial(ignore_logging, disable_upto=logging.NOTSET)],
}


@contextmanager
def algorithm_evaluator_verbosity(verbosity: int):
    if verbosity not in ALGORITHM_EVALUATOR_VERBOSITY_IGNORE:
        raise ValueError(
            f"Expected `verbosity` to be one of {list(ALGORITHM_EVALUATOR_VERBOSITY_IGNORE.keys())}; "
            f"found: {verbosity}"
        )
    ignore_fns: List[Callable] = ALGORITHM_EVALUATOR_VERBOSITY_IGNORE[verbosity]
    with ExitStack() as es:
        for ignore_fn in ignore_fns:
            es.enter_context(ignore_fn())
        yield
