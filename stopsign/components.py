"""
FastHTML Components
Reusable UI components for the Stop Sign application
"""

from fasthtml.common import H2
from fasthtml.common import H3
from fasthtml.common import H4
from fasthtml.common import A
from fasthtml.common import Button
from fasthtml.common import Div
from fasthtml.common import Footer
from fasthtml.common import Header
from fasthtml.common import Main
from fasthtml.common import Nav
from fasthtml.common import P
from fasthtml.common import Span


def video_component():
    """Video streaming component with reserved layout space"""
    return Div(
        Div(
            id="videoContainer",
            hx_get="/load-video",
            hx_trigger="load",
            cls="video-stream__player",
        ),
        cls="video-stream sunken",
    )


def recent_passes_component():
    """Recent vehicle passes component - loads once on page load"""
    return Div(
        H3("Recent Passes"),
        Div(
            # Initial load of recent passes - no auto updates
            hx_get="/api/recent-vehicle-passes",
            hx_trigger="load",
            hx_swap="innerHTML",
            id="recentPasses",
            cls="activity-feed",
        ),
        cls="window window--panel",
    )


def main_layout_component():
    """Main semantic layout for home page"""
    return Main(
        Div(
            Div(
                video_component(),
                live_stats_component(),
                cls="content-primary",
            ),
            Div(
                recent_passes_component(),
                cls="content-secondary",
            ),
            cls="content-grid",
        ),
        cls="app-layout",
    )


def page_head_component(title, include_video_deps=False, page_type="home"):
    """Standard page head with title and dependencies"""
    from fasthtml.common import Head
    from fasthtml.common import Link
    from fasthtml.common import Meta
    from fasthtml.common import Script
    from fasthtml.common import Title

    # Base URL for the site
    base_url = "https://crestwoodstopsign.com"

    # Page-specific descriptions and URLs
    page_metadata = {
        "home": {
            "description": (
                "Real-time AI traffic monitoring system using YOLO computer vision for stop sign "
                "compliance detection. Watch live vehicle detection and tracking at the Crestwood "
                "intersection with advanced traffic camera AI."
            ),
            "url": f"{base_url}/",
            "image": f"{base_url}/static/screenshot_afternoon.png",
        },
        "about": {
            "description": (
                "Learn about Stop Sign Nanny's real-time computer vision pipeline for AI traffic "
                "monitoring. Built with YOLO detection, FastHTML, and advanced stop sign compliance "
                "tracking technology."
            ),
            "url": f"{base_url}/about",
            "image": f"{base_url}/static/screenshot_afternoon.png",
        },
        "records": {
            "description": (
                "View vehicle records and stop sign compliance statistics from our AI traffic "
                "monitoring system. Browse historical data from real-time vehicle detection and "
                "YOLO computer vision analysis."
            ),
            "url": f"{base_url}/records",
            "image": f"{base_url}/static/screenshot_afternoon.png",
        },
        "debug": {
            "description": (
                "Debug interface for Stop Sign Nanny AI traffic monitoring system configuration " "and zone adjustment."
            ),
            "url": f"{base_url}/debug",
            "image": f"{base_url}/static/screenshot_afternoon.png",
        },
    }

    # Get metadata for current page type, fallback to home
    metadata = page_metadata.get(page_type, page_metadata["home"])

    scripts = [
        Script(src="https://unpkg.com/htmx.org@1.9.4", defer=True),
        Script(
            src="https://analytics.drose.io/script.js",
            defer=True,
            **{"data-website-id": "f5671ede-2232-44ea-9e5c-aabeeb766f95"},
        ),
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
        # Description meta tag
        Meta(name="description", content=metadata["description"]),
        # Canonical URL
        Link(rel="canonical", href=metadata["url"]),
        # Open Graph tags
        Meta(property="og:title", content=title),
        Meta(property="og:description", content=metadata["description"]),
        Meta(property="og:image", content=metadata["image"]),
        Meta(property="og:url", content=metadata["url"]),
        Meta(property="og:type", content="website"),
        # Twitter Card tags
        Meta(name="twitter:card", content="summary_large_image"),
        Meta(name="twitter:title", content=title),
        Meta(name="twitter:description", content=metadata["description"]),
        Meta(name="twitter:image", content=metadata["image"]),
        # Stylesheet
        Link(rel="stylesheet", href="/static/base.css"),
        *scripts,
    )


def common_header_component(title):
    """Common header with navigation"""
    return Header(
        Div(title, cls="title-bar"),
        Nav(
            Div(
                A("Home", href="/", cls="navigation__link"),
                A("Records", href="/records", cls="navigation__link"),
                A("About", href="/about", cls="navigation__link"),
                cls="navigation__group",
            ),
            Div(
                A("GitHub", href="https://github.com/cipher982/stopsign_ai", target="_blank", cls="navigation__link"),
                cls="navigation__group navigation__group--right",
            ),
            cls="navigation",
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
        Div(
            hx_get="/load-video",
            hx_trigger="load",
            cls="video-stream__player",
        ),
        cls="window window--panel",
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
                    cls="button button--zone active",
                ),
                Button(
                    "Pre-Stop",
                    id="zone-pre-stop",
                    onclick="selectZoneType('pre-stop')",
                    cls="button button--zone",
                ),
                Button(
                    "Capture",
                    id="zone-capture",
                    onclick="selectZoneType('capture')",
                    cls="button button--zone",
                ),
            ),
            P(id="zone-instructions"),
        ),
        # Visualization
        Div(
            H3("2. Visualization"),
            Button("Show Zones", id="debugZonesBtn", onclick="toggleDebugZones()", cls="button"),
        ),
        # Actions
        Div(
            H3("3. Edit Zone"),
            Button("Adjust", id="adjustmentModeBtn", onclick="toggleAdjustmentMode()", cls="button"),
            Button("Reset", onclick="resetPoints()", cls="button"),
            Button(
                "SUBMIT",
                id="submitBtn",
                onclick="updateZoneFromClicks()",
                style="display: none;",
                disabled=True,
                cls="button",
            ),
        ),
        # Status
        Div(
            H3("4. Status"),
            Div(id="config-version", cls="terminal terminal--version"),
            Div(id="status", cls="terminal terminal--status"),
            Div(
                P(
                    "Select zone → Show zones → Adjust → Click required points (4 for stop zone) → Submit",
                ),
            ),
        ),
        cls="debug-panel",
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
                    Span(cls="status-indicator status-indicator--good", id="systemStatus"),
                    Span(" Online"),
                ),
            ),
            # Rotating Insight
            Div(
                H4("Insight"),
                P("Loading...", id="rotatingInsight"),
            ),
            cls="metrics-grid",
        ),
        # Hidden div that triggers updates and distributes data
        Div(
            hx_get="/api/live-stats",
            hx_trigger="load, every 15s",
            hx_swap="none",
            hx_vals='{"update": "stats"}',
            id="statsUpdater",
        ),
        cls="window window--card",
    )


def debug_tools_component():
    """Debug tools section"""
    return Div(
        H3("Debug Tools"),
        Button("Coord Info", onclick="showCoordinateInfo()", cls="button"),
        Button("Debug Transforms", onclick="debugCoordinates()", cls="button"),
        Div(id="coordOutput", cls="terminal terminal--log"),
        Div(id="debugOutput", cls="terminal terminal--log"),
        cls="window window--panel",
    )
