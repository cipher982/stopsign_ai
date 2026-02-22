"""Vehicle pass scoring helpers (color indicators, grades)."""

COMPLIANCE_THRESHOLD_SECONDS = 2.0


def get_speed_color(speed):
    """Return CSS color for speed value."""
    if speed > 2.0:
        return "#ef4444"  # red-500
    elif speed > 1.5:
        return "#f97316"  # orange-500
    elif speed > 1.0:
        return "#eab308"  # yellow-500
    elif speed > 0.5:
        return "#84cc16"  # lime-500
    else:
        return "#22c55e"  # green-500


def get_time_color(time_val):
    """Return CSS color for time-in-zone value.

    Longer time = car stopped properly = green (good).
    Short time = blew through the stop sign = red (bad).
    """
    if time_val > 3.0:
        return "#22c55e"  # green-500 — good stop
    elif time_val > COMPLIANCE_THRESHOLD_SECONDS:
        return "#84cc16"  # lime-500 — adequate stop
    elif time_val > 1.5:
        return "#eab308"  # yellow-500 — insufficient stop
    elif time_val > 1.0:
        return "#f97316"  # orange-500 — barely slowed
    else:
        return "#ef4444"  # red-500 — ran the sign


def get_verdict_color(verdict: str) -> str:
    """Return CSS color for stop verdict label."""
    return {
        "Full Stop": "#22c55e",  # green-500
        "Rolling Stop": "#f59e0b",  # amber-500
        "No Stop": "#ef4444",  # red-500
    }.get(verdict, "#888888")


COLOR_MAP = {
    "white": "#e8e8e8",
    "black": "#333333",
    "silver": "#a0a0a0",
    "gray": "#808080",
    "grey": "#808080",
    "red": "#cc3333",
    "blue": "#3366cc",
    "green": "#339933",
    "brown": "#8b6914",
    "beige": "#d4c5a0",
    "gold": "#cca300",
    "yellow": "#cccc00",
    "orange": "#cc6600",
    "maroon": "#660000",
    "tan": "#c4a882",
}
