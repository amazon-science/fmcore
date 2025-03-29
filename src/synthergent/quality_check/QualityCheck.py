from abc import ABC, abstractmethod
from typing import *

import pandas as pd
from pydantic import ConfigDict, model_validator

from synthergent.constants import FileFormat
from synthergent.util import (
    DataFrameReader,
    Executor,
    ExecutorConfig,
    FileMetadata,
    Parameters,
    Step,
    String,
    as_list,
    as_set,
    safe_validate_arguments,
)


class QualityCheck(Step, ABC):
    class Params(Parameters):
        """
        BaseModel for parameters. Expected to be overridden by subclasses.
        """

        num_cpus: int = 1
        num_gpus: int = 0
        display_exclude: Tuple[str, ...] = ("num_cpus", "num_gpus")

        model_config = ConfigDict(
            extra="ignore",
        )

    params: Params = {}

    @model_validator(mode="before")
    @classmethod
    def _QualityCheck_convert_params(cls, params: Dict) -> Dict:
        params["params"] = cls._convert_params(cls.Params, params.get("params"))
        return params

    @abstractmethod
    def evaluate(
        self,
        data: pd.DataFrame,
        scaling: ExecutorConfig,
        executor: Optional[Executor],
        **kwargs,
    ) -> pd.DataFrame:
        pass

    @safe_validate_arguments
    def run(
        self,
        *,
        data: Any,
        scaling: ExecutorConfig,
        executor: Optional[Executor],
        step_i: int,
        num_steps: int,
        **kwargs,
    ) -> Dict:
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
        result: pd.DataFrame = self.evaluate(data=data, scaling=scaling, executor=executor, **kwargs)
        params_str: str = String.stringify(
            self.params.dict(
                exclude=as_list(["run_fn_spec", "input_aliases", "tracker", "verbosity", "display_exclude"])
                + as_list(self.params.display_exclude)
            )
        )
        output_key: str = f"""{self.class_name}({params_str})"""
        return {
            output_key: result,
        }
