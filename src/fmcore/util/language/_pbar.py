from typing import *

from pydantic import Extra, conint, root_validator
from pydantic.typing import Literal
from tqdm.auto import tqdm as AutoTqdmProgressBar
from tqdm.autonotebook import tqdm as NotebookTqdmProgressBar
from tqdm.std import tqdm as StdTqdmProgressBar

from ._alias import set_param_from_alias
from ._function import get_fn_spec
from ._import import _IS_RAY_INSTALLED, optional_dependency
from ._string import String
from ._structs import filter_keys, is_dict_like, is_list_or_set_like, remove_keys
from ._typing import MutableParameters, Parameters

TqdmProgressBar = Union[AutoTqdmProgressBar, NotebookTqdmProgressBar, StdTqdmProgressBar]

ProgressBar = "ProgressBar"


class ProgressBar(MutableParameters):
    pbar: Optional[TqdmProgressBar] = None
    style: Literal["auto", "notebook", "std", "ray"] = "auto"
    unit: str = "row"
    color: str = "#0288d1"  ## Bluish
    ncols: int = 100
    smoothing: float = 0.15
    total: Optional[int] = None
    disable: bool = False
    miniters: conint(ge=1) = 1
    _pending_updates: int = 0

    class Config(Parameters.Config):
        extra = Extra.allow

    @root_validator(pre=False)
    def _set_params(cls, params: Dict) -> Dict:
        set_param_from_alias(params, param="disable", alias=["disabled"])
        pbar: TqdmProgressBar = cls._create_pbar(**remove_keys(params, ["pbar", "color"]))
        pbar.color = params["color"]
        pbar.refresh()
        params["pbar"]: TqdmProgressBar = pbar
        return params

    @classmethod
    def _create_pbar(
        cls,
        style: Literal["auto", "notebook", "std", "ray"],
        **kwargs,
    ) -> TqdmProgressBar:
        if style == "auto":
            with optional_dependency("ipywidgets"):
                kwargs["ncols"]: Optional[int] = None
            return AutoTqdmProgressBar(**kwargs)
        elif style == "notebook":
            with optional_dependency("ipywidgets"):
                kwargs["ncols"]: Optional[int] = None
            return NotebookTqdmProgressBar(**kwargs)
        elif _IS_RAY_INSTALLED and style == "ray":
            from ray.experimental import tqdm_ray

            kwargs = filter_keys(
                kwargs,
                keys=set(get_fn_spec(tqdm_ray.tqdm).args + get_fn_spec(tqdm_ray.tqdm).kwargs),
                how="include",
            )
            return tqdm_ray.tqdm(**kwargs)
        else:
            return StdTqdmProgressBar(**kwargs)

    @classmethod
    def iter(cls, iterable: Union[Generator, Iterator, List, Tuple, Set, Dict, ItemsView], **kwargs):
        if is_list_or_set_like(iterable) or is_dict_like(iterable):
            kwargs["total"] = len(iterable)
        if is_dict_like(iterable):
            iterable: ItemsView = iterable.items()
        pbar: ProgressBar = ProgressBar.of(**kwargs)
        try:
            for item in iterable:
                yield item
                pbar.update(1)
            pbar.success()
        except Exception as e:
            pbar.failed()
            raise e

    @classmethod
    def of(
        cls,
        progress_bar: Optional[Union[ProgressBar, Dict, bool]] = True,
        *,
        prefer_kwargs: bool = True,
        **kwargs,
    ) -> ProgressBar:
        if isinstance(progress_bar, ProgressBar):
            if prefer_kwargs:
                if "total" in kwargs:
                    progress_bar.set_total(kwargs["total"])
                if "initial" in kwargs:
                    progress_bar.set_n(kwargs["initial"])
                if "desc" in kwargs:
                    progress_bar.set_description(kwargs["desc"])
                if "unit" in kwargs:
                    progress_bar.set_description(kwargs["unit"])
            return progress_bar
        if progress_bar is not None and not isinstance(progress_bar, (bool, dict)):
            raise ValueError(
                "You must pass `progress_bar` as either a bool, dict or None. None or False disables it."
            )
        if progress_bar is True:
            progress_bar: Optional[Dict] = dict()
        elif progress_bar is False:
            progress_bar: Optional[Dict] = None
        if progress_bar is not None and not isinstance(progress_bar, dict):
            raise ValueError(
                "You must pass `progress_bar` as either a bool, dict or None. None or False disables it."
            )
        if progress_bar is None:
            progress_bar: Dict = dict(disable=True)
        elif isinstance(progress_bar, dict) and len(kwargs) > 0:
            if prefer_kwargs is True:
                progress_bar: Dict = {
                    **progress_bar,
                    **kwargs,
                }
            else:
                progress_bar: Dict = {
                    **kwargs,
                    **progress_bar,
                }
        assert isinstance(progress_bar, dict)
        return ProgressBar(**progress_bar)

    def update(self, n: int = 1) -> Optional[bool]:
        self._pending_updates += n
        if abs(self._pending_updates) >= self.miniters:
            out = self.pbar.update(n=self._pending_updates)
            self.refresh()
            self._pending_updates = 0
            return out
        else:
            return None

    def set_n(self, new_n: int):
        self.pbar.update(n=new_n - self.pbar.n)
        self._pending_updates = 0  ## Clear all updates after setting new value
        self.refresh()

    def set_total(self, new_total: int):
        self.pbar.total = new_total
        self._pending_updates = 0  ## Clear all updates after setting new value
        self.refresh()

    def set_description(self, desc: Optional[str] = None, refresh: Optional[bool] = True):
        out = self.pbar.set_description(desc=desc, refresh=refresh)
        self.refresh()
        return out

    def set_unit(self, new_unit: str):
        self.pbar.unit = new_unit
        self.refresh()

    def success(self, desc: Optional[str] = None, close: bool = True, append_desc: bool = True):
        self._complete_with_status(
            color="#43a047",  ## Dark Green
            desc=desc,
            close=close,
            append_desc=append_desc,
        )

    def stopped(self, desc: Optional[str] = None, close: bool = True, append_desc: bool = True):
        self._complete_with_status(
            color="#b0bec5",  ## Dark Grey
            desc=desc,
            close=close,
            append_desc=append_desc,
        )

    def failed(self, desc: Optional[str] = None, close: bool = True, append_desc: bool = True):
        self._complete_with_status(
            color="#e64a19",  ## Dark Red
            desc=desc,
            close=close,
            append_desc=append_desc,
        )

    def _complete_with_status(
        self,
        color: str,
        desc: Optional[str],
        close: bool,
        append_desc: bool,
    ):
        if not self.pbar.disable:
            self.pbar.update(n=self._pending_updates)
            self._pending_updates = 0
            self.color = color
            self.pbar.colour = color
            if desc is not None:
                if append_desc:
                    desc: str = f"[{desc}] {self.pbar.desc}"
                self.pbar.desc = desc
            self.pbar.refresh()
            if close:
                self.close()

    def refresh(self):
        self.pbar.colour = self.color
        self.pbar.refresh()

    def close(self):
        self.pbar.refresh()
        self.pbar.close()
        self.pbar.refresh()

    def __del__(self):
        self.pbar.close()


def create_progress_bar(
    *,
    style: Optional[Literal["auto", "notebook", "std"]] = "auto",
    unit: str = "row",
    ncols: int = 100,
    smoothing: float = 0.1,
    **kwargs,
) -> TqdmProgressBar:
    try:
        if style == "auto":
            with optional_dependency("ipywidgets"):
                ncols: Optional[int] = None
            return AutoTqdmProgressBar(ncols=ncols, unit=unit, smoothing=smoothing, **kwargs)
        elif style == "notebook":
            with optional_dependency("ipywidgets"):
                ncols: Optional[int] = None
            return NotebookTqdmProgressBar(ncols=ncols, unit=unit, smoothing=smoothing, **kwargs)
        elif _IS_RAY_INSTALLED and style == "ray":
            from ray.experimental import tqdm_ray

            kwargs = filter_keys(
                kwargs,
                keys=set(get_fn_spec(tqdm_ray.tqdm).args + get_fn_spec(tqdm_ray.tqdm).kwargs),
                how="include",
            )
            return tqdm_ray.tqdm(**kwargs)
        else:
            return StdTqdmProgressBar(ncols=ncols, unit=unit, smoothing=smoothing, **kwargs)
    except Exception as e:
        kwargs["style"] = style
        kwargs["unit"] = unit
        kwargs["ncols"] = ncols
        kwargs["smoothing"] = smoothing
        raise ValueError(
            f"Error: could not create progress bar using settings: {kwargs}. Stack trace:\n{String.format_exception_msg(e)}"
        )
