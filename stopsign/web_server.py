# ruff: noqa: E501
import logging
import os
from datetime import datetime
from datetime import timedelta

import pytz
import uvicorn
from fasthtml.common import H1
from fasthtml.common import H2
from fasthtml.common import H3
from fasthtml.common import H4
from fasthtml.common import A
from fasthtml.common import Body
from fasthtml.common import Div
from fasthtml.common import FastHTML
from fasthtml.common import FileResponse
from fasthtml.common import Footer
from fasthtml.common import Head
from fasthtml.common import Header
from fasthtml.common import Html
from fasthtml.common import Iframe
from fasthtml.common import Img
from fasthtml.common import Li
from fasthtml.common import Link
from fasthtml.common import Main
from fasthtml.common import Nav
from fasthtml.common import P
from fasthtml.common import Script
from fasthtml.common import Span
from fasthtml.common import StaticFiles
from fasthtml.common import Style
from fasthtml.common import Title
from fasthtml.common import Ul
from fasthtml.common import Video

from stopsign.config import Config
from stopsign.database import Database

logger = logging.getLogger(__name__)


def get_env(key: str) -> str:
    value = os.getenv(key)
    assert value is not None, f"{key} is not set"
    return value


SQL_DB_NAME = get_env("SQL_DB_NAME")
SQL_DB_PATH = f"/app/data/{SQL_DB_NAME}"
GRAFANA_URL = get_env("GRAFANA_URL")
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080


def get_common_styles():
    return Style("""
            :root {
                --bg-color: #1a1a1a;
                --text-color: #e0e0e0;
                --accent-color: #ff6600;
                --secondary-color: #8b00ff;
                --card-bg: #2a2a2a;
            }
            body {
                font-family: 'Roboto', sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                line-height: 1.6;
            }
            header {
                background-color: var(--bg-color);
                border-bottom: 1px solid var(--accent-color);
            }
            h1, h2, h3 {
                color: var(--accent-color);
                text-shadow: 2px 2px 4px rgba(255, 102, 0, 0.5);
            }
            a {
                color: var(--secondary-color);
                text-decoration: none;
                transition: color 0.3s ease;
            }
            a:hover {
                color: var(--accent-color);
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px;
            }
            .card {
                background-color: var(--card-bg);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(255, 102, 0, 0.2);
                transition: transform 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
            }
            button {
                background-color: var(--accent-color);
                color: var(--bg-color);
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            button:hover {
                background-color: var(--secondary-color);
            }
            code {
                font-family: 'Roboto Mono', monospace;
                background-color: #3a3a3a;
                padding: 2px 4px;
                border-radius: 4px;
            }
            nav a {
                color: var(--accent-color);
                text-decoration: none;
                transition: color 0.3s ease;
                margin-left: 15px;
            }
            nav a:hover {
                color: var(--secondary-color);
            }
            ::-webkit-scrollbar {
                width: 10px;
            }

            ::-webkit-scrollbar-track {
                background: var(--bg-color);
                border-radius: 5px;
            }

            ::-webkit-scrollbar-thumb {
                background: var(--accent-color);
                border-radius: 5px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: var(--secondary-color);
            }

            /* For Firefox */
            * {
                scrollbar-width: thin;
                scrollbar-color: var(--accent-color) var(--bg-color);
            }
        """)


def get_common_header(title):
    return Header(
        H1(title, style="text-align: center; flex-grow: 1;"),
        Nav(
            A("Home", href="/"),
            A("Records", href="/records"),
            A("Statistics", href="/statistics"),
            A("About", href="/about"),
            A("GitHub", href="https://github.com/cipher982/stopsign_ai", target="_blank"),
            style="margin-left: 20px;",
        ),
        style="padding: 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--accent-color);",
    )


def get_common_footer():
    return Footer(
        P("By David Rose"),
        style="padding: 10px; text-align: center; border-top: 1px solid var(--accent-color);",
    )


# Initialize FastHTML app
app = FastHTML(
    ws_hdr=True,
    pico=False,  # We'll use our own styles instead of Pico CSS
    hdrs=(
        Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Roboto+Mono&display=swap",
        ),
        get_common_styles(),
        Script(src="https://unpkg.com/htmx.org@1.9.4"),
        # Script(src="https://unpkg.com/htmx.org/dist/ext/sse.js"),
    ),
)


app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/app/data", StaticFiles(directory="data"), name="data")
app.mount("/app/stream", StaticFiles(directory="/app/stream"), name="stream")


@app.get("/favicon.ico")  # type: ignore
async def favicon():
    return FileResponse("static/favicon.ico")


@app.get("/")  # type: ignore
def home():
    return Html(
        Head(
            Title("Stop Sign Nanny"),
            Script(src="https://unpkg.com/htmx.org@1.9.4"),
            Script(src="https://cdn.jsdelivr.net/npm/hls.js@latest"),
            Script("""
                function updateStopZone() {
                    const x1 = document.getElementById('x1').value;
                    const y1 = document.getElementById('y1').value;
                    const x2 = document.getElementById('x2').value;
                    const y2 = document.getElementById('y2').value;
                    
                    const points = [
                        {x: parseFloat(x1), y: parseFloat(y1)},
                        {x: parseFloat(x2), y: parseFloat(y2)}
                    ];
                    
                    fetch('/api/update-stop-zone', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({points: points}),
                    })
                    .then(response => response.json())
                    .then(data => {
                        console.log('Success:', data);
                        document.getElementById("status").innerText = 'Stop zone updated successfully!';
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                        document.getElementById("status").innerText = 'Failed to update stop zone.';
                    });
                }

                function fetchRecentPasses() {
                    fetch('/api/recent-vehicle-passes')
                        .then(response => {
                            if (!response.ok) {
                                throw new Error(`HTTP error! status: ${response.status}`);
                            }
                            return response.text();
                        })
                        .then(html => {
                            document.getElementById('recentPasses').innerHTML = html;
                        })
                        .catch(error => {
                            console.error('Error fetching recent passes:', error);
                            document.getElementById('recentPasses').innerHTML = `<p>Error fetching data: ${error.message}</p>`;
                        });
                }

                document.addEventListener('DOMContentLoaded', function() {
                    fetchRecentPasses();
                    setInterval(fetchRecentPasses, 30000);
                });
            """),
            Style("""
                .content-wrapper {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                .video-container {
                    position: relative;
                    width: 100%;
                }
                #videoFrame {
                    width: 100%;
                    height: auto;
                    border: 1px solid var(--accent-color);
                    border-radius: 8px;
                }
                #videoPlayer {
                    width: 100%;
                    height: auto;
                    border: 1px solid var(--accent-color);
                    border-radius: 8px;
                }
                #selectionCanvas {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    pointer-events: none;
                }
                .recent-passes {
                    width: 100%;
                    max-width: 500px;
                }
                @media (min-width: 768px) {
                    .content-wrapper {
                        flex-direction: row;
                    }
                    .video-container, .recent-passes {
                        flex: 1;
                    }
                }
            """),
        ),
        Body(
            get_common_header("Stop Sign Nanny"),
            Main(
                Div(
                    Div(
                        id="videoContainer",
                        hx_get="/load-video",
                        hx_trigger="load",
                    ),
                    cls="video-container",
                ),
                Div(
                    H2("Recent Vehicle Passes"),
                    Div(
                        Div(id="recentPasses"),
                        style="overflow-y: auto; max-height: 70vh;",
                    ),
                    cls="recent-passes",
                ),
                cls="content-wrapper",
            ),
            get_common_footer(),
            Script("""
                document.addEventListener('DOMContentLoaded', function() {
                    var video = document.getElementById('videoPlayer');
                    if (Hls.isSupported()) {
                        var hls = new Hls();
                        hls.loadSource('/app/stream/stream.m3u8');
                        hls.attachMedia(video);
                        hls.on(Hls.Events.MANIFEST_PARSED, function() {
                            video.play();
                        });
                    }
                    // HLS.js is not supported on platforms that do not have Media Source Extensions (MSE) enabled.
                    // When the browser has built-in HLS support (check using `canPlayType`), we can provide an HLS manifest (i.e. .m3u8 URL) directly to the video element through the `src` property.
                    // This is using the built-in support of the plain video element, without using HLS.js.
                    else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                        video.src = '/app/stream/stream.m3u8';
                        video.addEventListener('loadedmetadata', function() {
                            video.play();
                        });
                    }
                });
            """),
        ),
    )


@app.post("/api/update-stop-zone")  # type: ignore
async def update_stop_zone(request):
    data = await request.json()
    points = data["points"]
    stop_line = ((points[0]["x"], points[0]["y"]), (points[1]["x"], points[1]["y"]))
    stop_box_tolerance = 10
    min_stop_duration = 2.0

    new_config = {
        "stop_line": stop_line,
        "stop_box_tolerance": stop_box_tolerance,
        "min_stop_duration": min_stop_duration,
    }
    app.state.video_analyzer.reload_stop_zone_config(new_config)

    return {"status": "success"}


@app.get("/api/recent-vehicle-passes")  # type: ignore
async def get_recent_vehicle_passes():
    try:
        if not hasattr(app.state, "db"):
            app.state.db = Database(db_file=SQL_DB_PATH)
        recent_passes = app.state.db.get_recent_vehicle_passes(limit=50)

        local_tz = pytz.timezone("America/Chicago")

        def format_timestamp(timestamp_str):
            utc_time = datetime.fromisoformat(timestamp_str).replace(tzinfo=pytz.UTC)
            chicago_time = utc_time.astimezone(local_tz)
            now = datetime.now(local_tz)

            if chicago_time.date() == now.date():
                return chicago_time.strftime("%-I:%M:%S %p")
            elif chicago_time.date() == now.date() - timedelta(days=1):
                return f"Yesterday {chicago_time.strftime('%-I:%M %p')}"
            elif chicago_time.year == now.year:
                return chicago_time.strftime("%b %-d, %-I:%M %p")
            else:
                return chicago_time.strftime("%b %-d, %Y, %-I:%M %p")

        # Create a styled list of recent passes
        passes_list = Ul(
            *[
                Li(
                    Div(
                        Div(
                            Img(
                                src=pass_data[6],
                                alt="Vehicle Image",
                                style="width: 200px; height: auto; border-radius: 5px;",
                            ),
                            style="flex: 0 0 200px; margin-right: 15px;",
                        ),
                        Div(
                            Div(
                                Div(
                                    Span("Speed Score: ", style="font-weight: bold; font-size: 1.1em;"),
                                    Span(
                                        f"{calculate_speed_score(pass_data[5])}",
                                        style=f"font-weight: bold; font-size: 1.1em; color: hsl({calculate_speed_score(pass_data[5]) * 12}, 100%, 50%);",
                                    ),
                                ),
                                Div(
                                    f"Min Speed: {pass_data[5]:.2f} pixels/sec",
                                    style="font-size: 0.9em; margin-left: 10px;",
                                ),
                                style="margin-bottom: 10px;",
                            ),
                            Div(
                                Div(
                                    Span("Time Score: ", style="font-weight: bold; font-size: 1.1em;"),
                                    Span(
                                        f"{calculate_time_in_zone_score(pass_data[3])}",
                                        style=f"font-weight: bold; font-size: 1.1em; color: hsl({calculate_time_in_zone_score(pass_data[3]) * 12}, 100%, 50%);",
                                    ),
                                ),
                                Div(
                                    f"Time in Zone: {pass_data[3]:.2f} seconds",
                                    style="font-size: 0.9em; margin-left: 10px;",
                                ),
                                style="margin-bottom: 3px;",
                            ),
                            style="flex: 1;",
                        ),
                        style="display: flex; align-items: flex-start;",
                    ),
                    style="margin-bottom: 20px; background-color: var(--card-bg); padding: 15px; border-radius: 8px;",
                )
                for pass_data in recent_passes
            ],
            style="list-style-type: none; padding: 0; margin: 0;",
        )

        return Div(
            passes_list,
            id="recentPasses",
            style="background-color: var(--card-bg); padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,255,157,0.1);",
        )
    except Exception as e:
        logger.error(f"Error in get_recent_vehicle_passes: {str(e)}")
        return Div(P(f"Error: {str(e)}"), id="recentPasses")


@app.get("/records")  # type: ignore
def records():
    return Html(
        Head(
            Title("Records - Stop Sign Nanny"),
            Script(src="https://unpkg.com/htmx.org@1.9.4"),
        ),
        Body(
            get_common_header("Records"),
            get_common_styles(),
            Main(
                Div(
                    H2("Vehicle Records", style="text-align: center;"),
                    Div(
                        Div(id="worstPasses", hx_get="/api/worst-passes", hx_trigger="load"),
                        Div(id="bestPasses", hx_get="/api/best-passes", hx_trigger="load"),
                        style="display: flex; justify-content: center; gap: 20px;",
                    ),
                    cls="container",
                    style="margin: 20px auto; max-width: 1200px;",
                ),
            ),
            get_common_footer(),
        ),
    )


@app.get("/api/worst-passes")  # type: ignore
async def get_worst_passes():
    try:
        worst_speed_passes = app.state.db.get_extreme_passes("min_speed", "DESC", 5)
        worst_time_passes = app.state.db.get_extreme_passes("time_in_zone", "ASC", 5)
        return create_pass_list("Worst Passes", worst_speed_passes, worst_time_passes, "worstPasses")
    except Exception as e:
        logger.error(f"Error in get_worst_passes: {str(e)}")
        return Div(P(f"Error: {str(e)}"), id="worstPasses")


@app.get("/api/best-passes")  # type: ignore
async def get_best_passes():
    try:
        best_speed_passes = app.state.db.get_extreme_passes("min_speed", "ASC", 5)
        best_time_passes = app.state.db.get_extreme_passes("time_in_zone", "DESC", 5)
        return create_pass_list("Best Passes", best_speed_passes, best_time_passes, "bestPasses")
    except Exception as e:
        logger.error(f"Error in get_best_passes: {str(e)}")
        return Div(P(f"Error: {str(e)}"), id="bestPasses")


def create_pass_list(title, speed_passes, time_passes, div_id):
    return Div(
        H3(title),
        H4("Minimum Speed"),
        Div(*[create_pass_item(pass_data, "speed") for pass_data in speed_passes]),
        H4("Time in Stop Zone"),
        Div(*[create_pass_item(pass_data, "time") for pass_data in time_passes]),
        id=div_id,
        cls="card",
    )


def create_pass_item(pass_data, pass_type):
    # Determine if it's a best or worst pass
    is_best = (pass_type == "speed" and pass_data[5] < 10) or (pass_type == "time" and pass_data[3] > 3)

    # Set border color based on whether it's a best or worst pass
    border_color = "green" if is_best else "red"

    return Div(
        Div(
            Img(src=pass_data[6], alt="Vehicle Image", style="width: 200px; height: auto;"),
            style="display: inline-block; vertical-align: middle; margin-right: 10px;",
        ),
        Div(
            f"{pass_data[5]:.2f} pixels/sec" if pass_type == "speed" else f"{pass_data[3]:.2f} seconds",
            style="display: inline-block; vertical-align: middle;",
        ),
        style=f"""
            margin-bottom: 20px;
            text-align: left;
            border: 2px solid {border_color};
            border-radius: 8px;
            padding: 10px;
        """,
    )


@app.get("/statistics")  # type: ignore
def statistics():
    return Html(
        Head(
            Title("Statistics - Stop Sign Nanny"),
        ),
        Body(
            get_common_header("Statistics"),
            get_common_styles(),
            Main(
                Div(
                    Iframe(
                        src=GRAFANA_URL,
                        width="100%",
                        height="600",
                        frameborder="0",
                    ),
                    cls="container",
                ),
            ),
            get_common_footer(),
        ),
    )


@app.get("/about")  # type: ignore
def about():
    with open("static/summary.md", "r") as file:
        summary_content = file.read()
    return Html(
        Head(
            Title("About - Stop Sign Nanny"),
        ),
        Body(
            get_common_header("About"),
            get_common_styles(),
            Main(
                Div(
                    H2("Project Summary"),
                    P(summary_content),
                    cls="container",
                    style="margin: 20px auto; padding: 20px; background-color: var(--card-bg); border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 255, 157, 0.1); white-space: pre-wrap; max-width: 800px;",
                ),
            ),
            get_common_footer(),
        ),
    )


@app.get("/check-stream")  # type: ignore
async def check_stream():
    import os

    stream_path = "app/stream/stream.m3u8"
    if os.path.exists(stream_path):
        logger.info(f"Stream file exists: {stream_path}")
        with open(stream_path, "r") as f:
            content = f.read()
        logger.debug(f"Stream file content:\n{content}")
        return {"status": "exists", "content": content}
    else:
        logger.warning(f"Stream file does not exist: {stream_path}")
        return {"status": "not found"}


@app.get("/load-video")  # type: ignore
def load_video():
    return Div(
        Video(
            id="videoPlayer",
            controls=True,
            autoplay=True,
            muted=True,
        ),
        Script("""
            var video = document.getElementById('videoPlayer');
            if (Hls.isSupported()) {
                var hls = new Hls();
                hls.loadSource('/app/stream/stream.m3u8');
                hls.attachMedia(video);
                hls.on(Hls.Events.MANIFEST_PARSED, function() {
                    video.play();
                });
                hls.on(Hls.Events.ERROR, function(event, data) {
                    console.error('HLS error:', event, data);
                });
            } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                video.src = '/app/stream/stream.m3u8';
                video.addEventListener('loadedmetadata', function() {
                    video.play();
                });
            }
            video.addEventListener('error', function(e) {
                console.error('Video error:', e);
            });
        """),
    )


def calculate_time_in_zone_score(time_in_zone):
    percentile = app.state.db.get_time_in_zone_percentile(time_in_zone)
    return round(percentile / 10)


def calculate_speed_score(min_speed):
    percentile = app.state.db.get_min_speed_percentile(min_speed)
    return round((100 - percentile) / 10)


def main(config: Config):
    try:
        app.state.db = Database(db_file=SQL_DB_PATH)
        uvicorn.run(
            "stopsign.web_server:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="warning",
            # reload_dirs=["./stopsign"],  # Use a relative path
            # reload_excludes=["./app/data/*"],  # Use a relative path
        )
    except Exception as e:
        logger.error(f"Error in web server: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    config = Config("./config.yaml")
    main(config)
