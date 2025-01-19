from abc import ABC
from typing import *

import numpy as np

from fmcore.constants import SHORTHAND_TO_TENSOR_LAYOUT_MAP, DataLayout, MLType
from fmcore.data.reader.asset.AssetReader import AssetReader


class AudioReader(AssetReader, ABC):
    asset_mltype = MLType.AUDIO

    def _postprocess_asset(self, asset: Any, **kwargs) -> Any:
        asset: Any = super()._postprocess_asset(asset, **kwargs)
        if SHORTHAND_TO_TENSOR_LAYOUT_MAP[self.return_as] is DataLayout.NUMPY:
            if self.channels_first:
                asset: np.ndarray = np.moveaxis(asset, -1, 0)
        return asset
