import multiprocessing as mp
from typing import *

import pandas as pd
from pydantic import model_validator

from synthergent.constants import (
    FileFormat,
    Parallelize,
)
from synthergent.distillation.Distillation import Distillation
from synthergent.FinalStep import FinalStep
from synthergent.quality_check.QualityCheck import QualityCheck
from synthergent.util import (
    Chain,
    ChainExecution,
    ChainStep,
    DataFrameWriter,
    Executor,
    ExecutorConfig,
    FileMetadata,
    ScalableDataFrame,
    String,
    Writer,
    as_set,
    dispatch_executor,
    only_key,
    remove_keys,
    safe_validate_arguments,
    stop_executor,
    type_str,
)


class Synthergent(Chain):
    @model_validator(mode="before")
    @classmethod
    def _Synthergent_check_params(cls, params: Dict) -> Dict:
        num_steps: int = len(params["steps"])
        for step_i, step in enumerate(params["steps"]):
            if isinstance(step, ChainStep) and isinstance(step.chain, FinalStep):
                if step_i != num_steps - 1:
                    raise ValueError("When creating a FinalStep, it must be the last step.")
                for _nested_step in step.chain.steps:
                    if not isinstance(_nested_step, (QualityCheck, Distillation)):
                        raise ValueError(
                            f"Can only have QualityCheck and Distillation in FinalStep.of(); "
                            f"found: {type_str(_nested_step)}"
                        )
            if isinstance(step, QualityCheck):
                raise ValueError(
                    "Cannot have QualityCheck as an individual step in pipeline, it must be in FinalStep.of()"
                )
            if isinstance(step, Distillation):
                raise ValueError(
                    "Cannot have Distillation as an individual step in pipeline, it must be in FinalStep.of()"
                )
        return params

    @safe_validate_arguments
    def run(
        self,
        *args,
        scaling: ExecutorConfig = ExecutorConfig(
            partition_size=None,
            parallelize=Parallelize.sync,
            max_workers=max(1, min(mp.cpu_count() - 1, 16)),  ## Default: 1-6 processes
        ),
        batch_size: Optional[int] = None,
        save: Optional[Union[FileMetadata, Dict, str]] = None,
        return_data_on_save: bool = False,
        verbosity: int = 1,
        return_exn: bool = False,
        **kwargs,
    ) -> Optional[Union[pd.DataFrame, Dict, ChainExecution]]:
        kwargs["exn_name"] = "Synthergent"
        kwargs["background"] = False
        kwargs["scaling"] = scaling
        kwargs["verbosity"] = verbosity

        if save is not None:
            save: FileMetadata = FileMetadata.of(save)
            available_df_writers: Set[FileFormat] = set()
            for df_writer_cls in DataFrameWriter.subclasses():
                assert issubclass(df_writer_cls, DataFrameWriter)
                available_df_writers |= as_set(df_writer_cls.file_formats)
            available_df_writers_str: str = String.join_human([x.lower() for x in available_df_writers])
            if save.format is None:
                raise ValueError(
                    "Unable to determine format in which to save data.\n"
                    '- If passing a folder path, please pass a dict as: {"path": "/path/to/data/", "format": "parquet"}\n'
                    '- If passing a file path, please pass a dict as: {"path": "/path/to/data.parquet", "format": "parquet"}\n'
                    f"Available formats are: {available_df_writers_str}"
                )
            elif save.format not in available_df_writers:
                raise ValueError(
                    f"Unsupported format to write data: {save.format}\n"
                    f"Available formats are: {available_df_writers_str}"
                )

        executor: Optional[Executor] = None
        try:
            if scaling.parallelize in {Parallelize.sync, Parallelize.threads, Parallelize.processes}:
                ## Run locally:
                if executor is None:
                    executor: Optional[Executor] = dispatch_executor(
                        parallelize=scaling.parallelize,
                        max_workers=scaling.max_workers,
                    )
            elif scaling.parallelize in {Parallelize.ray}:
                ## Run using Dask-on-Ray:
                pass
            else:
                raise NotImplementedError(f"Unsupported value: `scaling` = {scaling}")

            kwargs["executor"] = executor
            exn: ChainExecution = super(Synthergent, self).run(*args, **kwargs)
            if save is not None:
                writer: Writer = Writer.of(
                    save.format,
                    num_rows={Parallelize.ray: None}.get(scaling.parallelize, batch_size),
                )
                writer.write(
                    data=exn.outputs["data"],
                    destination=save,
                    file_name="part",
                )
                if return_data_on_save is False:
                    print(f'Saved to: "{save.path}"')
                    if return_exn:
                        return exn
                    else:
                        return None
            if return_exn:
                return exn
            outputs = {}
            if "final_step_results" in exn.outputs:
                outputs = {
                    **remove_keys(exn.outputs, ["final_step_results"]),
                    **exn.outputs["final_step_results"],
                }
            outputs["data"]: pd.DataFrame = ScalableDataFrame.of(outputs["data"]).pandas()
            if len(outputs) == 1 and only_key(outputs) == "data":
                return outputs["data"]
            return outputs
        finally:
            stop_executor(executor)
