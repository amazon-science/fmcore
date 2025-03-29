from typing import *

import pandas as pd

from synthergent.constants import FileFormat
from synthergent.distillation.Distillation import Distillation
from synthergent.quality_check.QualityCheck import QualityCheck
from synthergent.util import (
    ChainExecution,
    DataFrameReader,
    Executor,
    FileMetadata,
    ParallelMap,
    ScalableDataFrame,
    String,
    as_set,
    safe_validate_arguments,
    type_str,
    ExecutorConfig,
)


class FinalStep(ParallelMap):
    combine: Literal["list", "dict", "merge"] = "list"
    output_key: str = "final_step_results"

    @safe_validate_arguments
    def run(
        self,
        *,
        data: Any,
        scaling: ExecutorConfig,
        executor: Optional[Executor],
        step_i: int,
        num_steps: int,
        verbosity: int,
        **kwargs,
    ) -> ChainExecution:
        kwargs["exn_name"] = "FinalStep"
        kwargs["background"] = False
        kwargs["scaling"] = scaling
        kwargs["verbosity"] = verbosity

        if isinstance(data, (str, dict, FileMetadata)):
            data: FileMetadata = FileMetadata.of(data)

            available_df_readers: Set[FileFormat] = set()
            for df_reader_cls in DataFrameReader.subclasses():
                assert issubclass(df_reader_cls, DataFrameReader)
                available_df_readers |= as_set(df_reader_cls.file_formats)
            available_df_readers_str: str = String.join_human([x.lower() for x in available_df_readers])
            if data.format is None:
                raise ValueError(
                    "Unable to determine format for data. "
                    '\n- If passing a folder path, please pass a dict as: {"path": "/path/to/data/", "format": "parquet"}'
                    '\n- If passing a file path, please pass a dict as: {"path": "/path/to/data.parquet", "format": "parquet"}'
                    f"\n Available formats are: {available_df_readers_str}"
                )
            elif data.format not in available_df_readers:
                raise ValueError(
                    f"Unsupported format to read data: {data.format}\n"
                    f"Available formats are: {available_df_readers_str}"
                )
        # if scaling.parallelize in {Parallelize.ray} and executor is None:
        #     ## Run using Dask-on-Ray:
        #     executor: Executor = dispatch_executor(
        #         config=ExecutorConfig(
        #             parallelize=scaling.parallelize,
        #             max_workers=scaling.max_workers,
        #         )
        #     )
        data: pd.DataFrame = ScalableDataFrame.of(data).pandas()

        kwargs["executor"] = executor
        final_step_exn: ChainExecution = super(ParallelMap, self).run(
            data=data,
            **kwargs,
        )
        final_outputs: Dict = {
            "data": data,
        }
        for step_i, (step, step_outputs) in enumerate(
            zip(self.steps, final_step_exn.outputs[self.output_key])
        ):
            if isinstance(step, QualityCheck):
                final_outputs.setdefault("quality_checks", {})
                final_outputs["quality_checks"] = {
                    **final_outputs["quality_checks"],
                    **step_outputs,
                }
            elif isinstance(step, Distillation):
                final_outputs.setdefault("distillation", {})
                final_outputs["distillation"] = {
                    **final_outputs["distillation"],
                    **step_outputs,
                }
            else:
                raise ValueError(f"Unexpected: {type_str(step)} in FinalStep.")
        final_step_exn.outputs[self.output_key] = final_outputs
        return final_step_exn
