from abc import ABC, abstractmethod
from typing import *

import pandas as pd
from pydantic import ConfigDict, model_validator

from synthergent.constants import DataLayout, FileFormat, Parallelize
from synthergent.util import (
    DataFrameReader,
    FileMetadata,
    Parameters,
    Reader,
    ScalableDataFrame,
    Step,
    String,
    Timer,
    accumulate,
    as_set,
    dispatch,
    get_default,
    resolve_sample_size,
    safe_validate_arguments,
    ExecutorConfig,
)


class Cleaner(Step, ABC):
    class Params(Parameters):
        """
        BaseModel for parameters. Expected to be overridden by subclasses.
        """

        persist: bool = False

        model_config = ConfigDict(
            extra="forbid",
        )

    params: Params = {}

    @model_validator(mode="before")
    @classmethod
    def _Cleaner_convert_params(cls, params: Dict) -> Dict:
        params["params"] = cls._convert_params(cls.Params, params.get("params"))
        return params

    @abstractmethod
    def clean(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        pass

    @safe_validate_arguments
    def run(
        self,
        *,
        data: Any,
        scaling: ExecutorConfig,
        executor: Optional[Any],
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
        data: ScalableDataFrame = self._read_data(
            data,
            scaling=scaling,
            verbosity=self.verbosity,
        )
        if self.class_name == "SubSample":
            data: ScalableDataFrame = ScalableDataFrame.of(data)
            if data.layout is DataLayout.DASK:
                if self.params.persist:
                    data: ScalableDataFrame = data.persist(wait=True)
                data: ScalableDataFrame = data.sample(
                    frac=resolve_sample_size(self.params.size, length=len(data)) / len(data),
                    random_state=self.params.seed,
                )
            else:
                data: ScalableDataFrame = data.sample(
                    n=resolve_sample_size(self.params.size, length=len(data)),
                    random_state=self.params.seed,
                )
        elif scaling.parallelize in {Parallelize.sync, Parallelize.threads, Parallelize.processes}:
            ## Run locally:
            data: ScalableDataFrame = self._clean_local(
                data,
                scaling=scaling,
                executor=executor,
                verbosity=self.verbosity,
            )
            data: pd.DataFrame = data.pandas()
        elif scaling.parallelize in {Parallelize.ray}:
            ## Run using Dask-on-Ray:
            data: ScalableDataFrame = self._clean_dask(
                data,
                step_i=step_i,
                num_steps=num_steps,
                verbosity=self.verbosity,
            )
        else:
            raise NotImplementedError(f"Unsupported parameter: `scaling` = {scaling}")

        return {
            "data": data,
        }

    def _read_data(
        self,
        data: Union[pd.DataFrame, FileMetadata],
        *,
        scaling: ExecutorConfig,
        verbosity: int,
    ) -> ScalableDataFrame:
        silent: bool = {0: True, 1: True}.get(verbosity, False)
        if scaling.parallelize in {Parallelize.sync, Parallelize.threads, Parallelize.processes}:
            ## Read locally:
            if isinstance(data, FileMetadata):
                with Timer("Reading data", silent=silent):
                    return Reader.of(data.format).read(data, layout=DataLayout.PANDAS)
            else:
                return ScalableDataFrame.of(data, layout=DataLayout.PANDAS)
        elif scaling.parallelize in {Parallelize.ray}:
            if isinstance(data, FileMetadata):
                ## Read using Dask-on-Ray:
                with Timer("Reading data using Dask-on-Ray", silent=silent):
                    data: ScalableDataFrame = Reader.of(data.format).read(
                        data,
                        layout=DataLayout.DASK,
                    )
            else:
                with Timer("Converting data to Dask-on-Ray dataframe", silent=silent):
                    data: ScalableDataFrame = ScalableDataFrame.of(data)
                    if data.layout is not DataLayout.DASK:
                        data: ScalableDataFrame = data.to_layout(DataLayout.DASK)
            if scaling.partition_size is not None:
                data: ScalableDataFrame = data.repartition(
                    partition_size=scaling.partition_size,
                )
            assert isinstance(data, ScalableDataFrame)
            assert data.layout is DataLayout.DASK
            return data
        else:
            raise NotImplementedError(f"Unsupported value: `scaling` = {scaling}")

    def _clean_local(
        self,
        data: ScalableDataFrame,
        *,
        scaling: ExecutorConfig,
        executor: Optional[Any],
        verbosity: int,
    ) -> ScalableDataFrame:
        data: ScalableDataFrame = ScalableDataFrame.of(data, layout=DataLayout.PANDAS)
        futs = []
        for data_batch in data.stream(batch_size=get_default(scaling.batch_size, len(data))):
            futs.append(
                dispatch(
                    self.clean,
                    data_batch,
                    parallelize=scaling.parallelize,
                    executor=executor,
                )
            )
        return ScalableDataFrame.concat(
            accumulate(
                futs,
                progress_bar={0: False, 1: False}.get(verbosity, True),
            )
        )

    def _clean_dask(
        self,
        data: ScalableDataFrame,
        *,
        step_i: int,
        num_steps: int,
        verbosity: int,
    ) -> ScalableDataFrame:
        silent: bool = {0: True, 1: True}.get(verbosity, False)
        with Timer("Running data using Dask-on-Ray", silent=silent):
            if step_i == 0:
                ## First step, persist data:
                data = data.persist(wait=True)
            data = data.map_partitions(
                self.clean,
            )
            if self.params.persist:
                data = data.persist(wait=True)
            if step_i == num_steps - 1:
                ## Final step, compute result:
                data = data.compute()
            return data
