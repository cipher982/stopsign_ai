"""Lookout: turn a camera into a watch you can hand a job to.

Public surface used by the analyzer, the web app, and the replay harness:

* :class:`LookoutManager` — owns the bounded worker, live state, evidence, and
  delivery. The analyzer only ever calls ``submit_frame`` from its hot path.
* :func:`arm_watch` — build a watch from an exact arming frame plus a box.
* ``replay`` drives the same manager offline over a recorded sequence.
"""

from stopsign.lookout.manager import LookoutManager
from stopsign.lookout.manager import arm_watch
from stopsign.lookout.options import LookoutOptions
from stopsign.lookout.types import Condition
from stopsign.lookout.types import LookoutEvent
from stopsign.lookout.types import Watch
from stopsign.lookout.types import WatchKind
from stopsign.lookout.types import WatchState

__all__ = [
    "Condition",
    "LookoutEvent",
    "LookoutManager",
    "LookoutOptions",
    "Watch",
    "WatchKind",
    "WatchState",
    "arm_watch",
]
