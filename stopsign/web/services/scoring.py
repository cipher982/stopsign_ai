"""Vehicle pass scoring helpers (color indicators, grades)."""

COMPLIANCE_THRESHOLD_SECONDS = 2.0


def get_speed_color(speed):
    """Return CSS color (design-system token) for speed value."""
    if speed > 1.5:
        return "var(--bad)"
    elif speed > 1.0:
        return "var(--warn)"
    else:
        return "var(--ok)"


def get_time_color(time_val):
    """Return CSS color (design-system token) for time-in-zone value.

    Longer time = car stopped properly = green (good).
    Short time = blew through the stop sign = red (bad).
    """
    if time_val > COMPLIANCE_THRESHOLD_SECONDS:
        return "var(--ok)"
    elif time_val > 1.0:
        return "var(--warn)"
    else:
        return "var(--bad)"


def get_verdict_color(verdict: str) -> str:
    """Return CSS color (design-system token) for stop verdict label."""
    return {
        "Full Stop": "var(--ok)",
        "Rolling Stop": "var(--warn)",
        "No Stop": "var(--bad)",
    }.get(verdict, "var(--text-dim)")


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


def get_color_hex(name: str) -> str:
    """Map a free-form color label to a swatch hex, handling compounds like "dark gray"."""
    n = name.lower().strip()
    if n in COLOR_MAP:
        return COLOR_MAP[n]
    mapped = [w for w in n.split() if w in COLOR_MAP]
    if mapped:
        return COLOR_MAP[mapped[-1]]
    return "#55585f"
