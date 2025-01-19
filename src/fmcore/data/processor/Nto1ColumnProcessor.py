from abc import ABC, abstractmethod
from typing import *

from fmcore.constants import DataLayout
from fmcore.data.processor import DataProcessor
from fmcore.data.sdf import ScalableDataFrame, ScalableOrRaw, ScalableSeries, ScalableSeriesOrRaw, is_scalable
from fmcore.util import safe_validate_arguments


class Nto1ColumnProcessor(DataProcessor, ABC):
    """Abstract base class for N:1 data processors."""

    @safe_validate_arguments
    def fit(
        self,
        data: ScalableOrRaw,
        process_as: Optional[DataLayout] = None,
    ):
        data: ScalableDataFrame = ScalableDataFrame.of(data, layout=process_as)
        self._fit_df(data)

    def _fit_df(self, data: ScalableDataFrame):
        """Fit step is a noop by default."""
        pass

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @safe_validate_arguments
    def transform(
        self,
        data: ScalableOrRaw,
        process_as: Optional[DataLayout] = None,
    ) -> ScalableSeriesOrRaw:
        output_data: ScalableSeries = self._transform_df(ScalableDataFrame.of(data, layout=process_as))
        if is_scalable(data):
            return output_data
        return output_data.raw()

    @abstractmethod
    def _transform_df(self, data: ScalableDataFrame) -> ScalableSeries:
        """N:1 data processors can make optimizations internally as column-wise operations are usually much faster."""
        pass

    @safe_validate_arguments
    def fit_transform(
        self,
        data: ScalableOrRaw,
        process_as: Optional[DataLayout] = None,
    ) -> ScalableSeries:
        self.fit(data, process_as=process_as)
        return self.transform(data, process_as=process_as)
