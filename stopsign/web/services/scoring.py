"""Vehicle pass scoring helpers (color indicators, grades)."""


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
    """Return CSS color for time-in-zone value."""
    if time_val > 4.0:
        return "#ef4444"
    elif time_val > 3.0:
        return "#f97316"
    elif time_val > 2.0:
        return "#eab308"
    elif time_val > 1.0:
        return "#84cc16"
    else:
        return "#22c55e"


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
