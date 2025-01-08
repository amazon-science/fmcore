from fmcore.data.sdf.ScalableSeries import *
from fmcore.data.sdf.TensorScalableSeries import *
from fmcore.data.sdf.ScalableDataFrame import *
from fmcore.data.sdf.NumpyArrayScalableSeries import *
from fmcore.data.sdf.TorchScalableSeries import *
from fmcore.data.sdf.DatumScalableSeries import *
from fmcore.data.sdf.PandasScalableSeries import *
from fmcore.data.sdf.DaskScalableSeries import *
from fmcore.data.sdf.RecordScalableDataFrame import *
from fmcore.data.sdf.ListOfDictScalableDataFrame import *
from fmcore.data.sdf.DictScalableDataFrame import *
from fmcore.data.sdf.PandasScalableDataFrame import *
from fmcore.data.sdf.DaskScalableDataFrame import *

#
# def __get_dataframe_reader(DataFrameReaderClass, **kwargs):
#     instance_kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in DataFrameReaderClass.properties()}
#     instance_kwargs.pop('params', None)
#     params: Dict[str, Any] = {k: v for k, v in kwargs.items() if k not in DataFrameReaderClass.properties()}
#     return DataFrameReaderClass(params=params, **instance_kwargs)
#
#
# def read_csv(source: Any, **kwargs) -> ScalableDataFrameOrRaw:
#     from fmcore.data.reader import CsvReader
#     return __get_dataframe_reader(CsvReader, **kwargs).read(source, **kwargs)
#
#
# def read_parquet(source: Any, **kwargs) -> ScalableDataFrameOrRaw:
#     from fmcore.data.reader import ParquetReader
#     return __get_dataframe_reader(ParquetReader, **kwargs).read(source, **kwargs)
#
#
# def read_json(source: Any, **kwargs) -> ScalableDataFrameOrRaw:
#     from fmcore.data.reader import JsonLinesReader
#     return __get_dataframe_reader(JsonLinesReader, **kwargs).read(source, **kwargs)
