import warnings
from fmcore.framework.tracker.Tracker import *
from fmcore.framework.tracker.AimTracker import *
from fmcore.framework.tracker.LogFileTracker import *

DEFAULT_TRACKER: Optional[Tracker] = None

try:
    from fmcore.util.language import get_default
    from fmcore.util.jupyter import JupyterNotebook
    from fmcore.util.environment import EnvUtil

    if JupyterNotebook.is_notebook() and bool(get_default(EnvUtil.get_var('ENABLE_DEFAULT_TRACKER', False))):
        DEFAULT_TRACKER: Tracker = Tracker.default()
except Exception as e:
    warnings.warn(
        f'Cannot capture automatic logs using tracker: {DEFAULT_TRACKER_PARAMS}.'
        f'\nFollowing error was thrown: {str(e)}'
    )
