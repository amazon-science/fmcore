import warnings
from fmcore.framework._tracker.Tracker import *
from fmcore.framework._tracker.AimTracker import *
from fmcore.framework._tracker.LogFileTracker import *

DEFAULT_TRACKER: Optional[Tracker] = None

try:
    from bears.util.language import get_default
    from bears.util.jupyter import JupyterNotebook
    from bears.util.environment import EnvUtil

    if JupyterNotebook.is_notebook() and bool(get_default(EnvUtil.get_var('ENABLE_DEFAULT_TRACKER', False))):
        DEFAULT_TRACKER: Tracker = Tracker.default()
except Exception as e:
    warnings.warn(
        f'Cannot capture automatic logs using tracker: {DEFAULT_TRACKER_PARAMS}.'
        f'\nFollowing error was thrown: {str(e)}'
    )
