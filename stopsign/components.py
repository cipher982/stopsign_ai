"""
FastHTML Components
Reusable UI components for the Stop Sign application
"""

from fasthtml.common import H2
from fasthtml.common import H3
from fasthtml.common import H4
from fasthtml.common import A
from fasthtml.common import Button
from fasthtml.common import Details
from fasthtml.common import Div
from fasthtml.common import Footer
from fasthtml.common import Header
from fasthtml.common import Input
from fasthtml.common import Label
from fasthtml.common import Main
from fasthtml.common import Nav
from fasthtml.common import P
from fasthtml.common import Span
from fasthtml.common import Summary


def video_component():
    """Video streaming component with reserved layout space"""
    return Div(
        Div(
            id="videoContainer",
            hx_get="/load-video",
            hx_trigger="load once",
        ),
        cls="sunken",
    )


def recent_passes_component():
    """Recent vehicle passes component - no auto refresh to prevent image reloading"""
    return Div(
        H2("Recent Vehicle Passes"),
        Div(id="recentPasses"),
        cls="window",
    )


def main_layout_component():
    """Main two-column layout for home page"""
    return Main(
        Div(
            Div(
                video_component(),
                live_stats_component(),
            ),
            recent_passes_component(),
            cls="two-col",
        ),
        adjustment_panel_component(),
    )


def adjustment_panel_component():
    """Stop line adjustment panel (hidden by default)"""
    return Div(
        H3("Stop Line Adjustment"),
        # Click-to-set interface
        Div(
            Button(
                "Adjust Stop Line",
                id="adjustmentModeBtn",
                onclick="toggleAdjustmentMode()",
            ),
            P(
                "Click the button above, then click two points on the video to set the new stop line position.",
            ),
        ),
        # Manual coordinate input (legacy interface)
        Details(
            Summary("Manual Coordinate Input"),
            Div(
                Div(
                    Label("Point 1 - X:"),
                    Input(type="number", id="x1", value="550"),
                    Label("Y:"),
                    Input(type="number", id="y1", value="500"),
                ),
                Div(
                    Label("Point 2 - X:"),
                    Input(type="number", id="x2", value="400"),
                    Label("Y:"),
                    Input(type="number", id="y2", value="550"),
                ),
                Button(
                    "Update Stop Zone",
                    onclick="updateStopZone()",
                ),
                cls="sunken",
            ),
        ),
        # Status display
        Div(id="status"),
        id="adjustmentPanel",
        style="display: none;",
        cls="window",
    )


def page_head_component(title, include_video_deps=False, page_type="home"):
    """Standard page head with title and dependencies"""
    from fasthtml.common import Head
    from fasthtml.common import Link
    from fasthtml.common import Meta
    from fasthtml.common import Script
    from fasthtml.common import Title

    scripts = [
        Script(src="https://unpkg.com/htmx.org@1.9.4", defer=True),
    ]

    if include_video_deps:
        scripts.append(Script(src="https://cdn.jsdelivr.net/npm/hls.js@latest", defer=True))
        # Add video player JavaScript
        scripts.append(Script(src="/static/js/video-player.js", defer=True))

    # Add page-specific JavaScript
    if page_type == "home":
        scripts.append(Script(src="/static/js/home.js", defer=True))
    elif page_type == "debug":
        scripts.append(Script(src="/static/js/debug.js", defer=True))

    return Head(
        Title(title),
        Meta(name="viewport", content="width=device-width, initial-scale=1"),
        Link(rel="stylesheet", href="/static/base.css"),
        *scripts,
    )


def common_header_component(title):
    """Common header with navigation"""
    return Header(
        Div(title, cls="title-bar"),
        Nav(
            A("Home", href="/"),
            A("Records", href="/records"),
            A("About", href="/about"),
            A("GitHub", href="https://github.com/cipher982/stopsign_ai", target="_blank"),
        ),
        cls="window",
    )


def common_footer_component():
    """Common footer"""
    return Footer(
        P("By David Rose"),
        cls="window",
    )


def debug_video_component():
    """Video component for debug page"""
    return Div(
        H2("Video Stream"),
        Div(hx_get="/load-video", hx_trigger="load"),
        cls="debug-card",
    )


def debug_controls_component():
    """Debug page control panel"""
    return Div(
        # Zone Selection
        Div(
            H3("1. Select Zone"),
            Div(
                Button(
                    "Stop Line",
                    id="zone-stop-line",
                    onclick="selectZoneType('stop-line')",
                    cls="zone-selector active",
                ),
                Button(
                    "Pre-Stop",
                    id="zone-pre-stop",
                    onclick="selectZoneType('pre-stop')",
                    cls="zone-selector",
                ),
                Button(
                    "Capture",
                    id="zone-capture",
                    onclick="selectZoneType('capture')",
                    cls="zone-selector",
                ),
            ),
            P(id="zone-instructions"),
        ),
        # Visualization
        Div(
            H3("2. Visualization"),
            Button("Show Zones", id="debugZonesBtn", onclick="toggleDebugZones()"),
        ),
        # Actions
        Div(
            H3("3. Edit Zone"),
            Button("Adjust", id="adjustmentModeBtn", onclick="toggleAdjustmentMode()"),
            Button("Reset", onclick="resetPoints()"),
            Button(
                "SUBMIT",
                id="submitBtn",
                onclick="updateZoneFromClicks()",
                style="display: none;",
                disabled=True,
            ),
        ),
        # Status
        Div(
            H3("4. Status"),
            Div(id="status"),
            Div(
                P(
                    "Select zone → Show zones → Adjust → Click 2 points → Submit",
                ),
            ),
        ),
        cls="debug-card",
    )


def live_stats_component():
    """Dense live statistics panel - Win98 Task Manager style"""
    return Div(
        Div("System Status", cls="title-bar"),
        Div(
            # Today's Report Card
            Div(
                H4("Today's Report"),
                Div(
                    Span("Compliance: "),
                    Span("...", id="complianceRate"),
                ),
                Div(
                    Span("Violations: "),
                    Span("...", id="violationCount"),
                ),
                Div(
                    Span("Trend: "),
                    Span("→", id="trendArrow"),
                ),
            ),
            # Live Activity
            Div(
                H4("Activity"),
                Div(
                    Span("Vehicles: "),
                    Span("...", id="vehicleCount"),
                ),
                Div(
                    Span("Last: "),
                    Span("...", id="lastDetection"),
                ),
                Div(
                    Span("Status: "),
                    Span(cls="status-indicator good", id="systemStatus"),
                    Span(" Online"),
                ),
            ),
            # Rotating Insight
            Div(
                H4("Insight"),
                P("Loading...", id="rotatingInsight"),
            ),
            cls="three-col",
        ),
        # Hidden div that triggers updates and distributes data
        Div(
            hx_get="/api/live-stats",
            hx_trigger="load, every 30s",
            hx_swap="none",
            hx_vals='{"update": "stats"}',
            id="statsUpdater",
        ),
        cls="metric-card",
    )


def debug_tools_component():
    """Debug tools section"""
    return Div(
        H3("Debug Tools"),
        Button("Coord Info", onclick="showCoordinateInfo()"),
        Button("Debug Transforms", onclick="debugCoordinates()"),
        Div(id="coordOutput"),
        Div(id="debugOutput"),
        cls="window",
    )
