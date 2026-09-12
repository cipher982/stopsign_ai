"""The analyzer stall watchdog must tell starvation from a wedge.

Restarting the process cannot fix a dead camera, and on 2026-09-12 it tried anyway:
280 restarts in ten hours, each reloading the model against a feed that was not there.
The distinction is whether work is waiting - frames queued while nothing is processed
is the wedge the watchdog exists for; an empty queue is upstream.
"""

from unittest.mock import MagicMock

from stopsign.video_analyzer import VideoAnalyzer


def _analyzer(pending=None, raises=False):
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.redis_client = MagicMock()
    if raises:
        analyzer.redis_client.llen.side_effect = RuntimeError("redis down")
    else:
        analyzer.redis_client.llen.return_value = pending
    return analyzer


def test_an_empty_queue_is_starvation_not_a_stall():
    assert _analyzer(pending=0)._stall_should_exit(lag=600.0) is False


def test_frames_waiting_while_nothing_is_processed_is_a_stall():
    analyzer = _analyzer(pending=42)

    assert analyzer._stall_should_exit(lag=600.0) is True
    analyzer.redis_client.set.assert_called_once()
    recorded = analyzer.redis_client.set.call_args[0][1]
    assert "42 frame" in recorded


def test_an_unreadable_queue_keeps_the_original_behaviour():
    """Not knowing the depth is not a reason to assume there is no work."""
    assert _analyzer(raises=True)._stall_should_exit(lag=600.0) is True


def test_a_starvation_wait_does_not_write_a_stall_reason():
    analyzer = _analyzer(pending=0)

    analyzer._stall_should_exit(lag=600.0)

    analyzer.redis_client.set.assert_not_called()
