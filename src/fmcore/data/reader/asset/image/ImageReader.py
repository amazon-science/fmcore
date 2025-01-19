import io
from abc import ABC, abstractmethod
from typing import *

from pydantic.typing import Literal

from fmcore.constants import FileContents, MLType, Storage
from fmcore.data.asset import Image
from fmcore.data.reader.asset.AssetReader import AssetReader


class ImageReader(AssetReader, ABC):
    asset_mltype = MLType.IMAGE

    ## Whether to put the color channels first, i.e. get a 512x512 image as an array/tensor of shape (3, 512, 512)
    channels: Literal["first", "last"] = "first"

    def _read_asset(self, source: Union[str, io.BytesIO], **kwargs) -> Image:
        return self._read_image(source=source, **kwargs)

    @abstractmethod
    def _read_image(
        self,
        source: Union[str, io.BytesIO],
        storage: Storage,
        file_contents: Optional[FileContents] = None,
        **kwargs,
    ) -> Image:
        pass
