# ruff: noqa: E501
"""SEO: JSON-LD, page metadata, and structured data."""

BASE_URL = "https://crestwoodstopsign.com"

PAGE_METADATA = {
    "home": {
        "description": "Real-time AI traffic monitoring system using YOLO computer vision for stop sign compliance detection. Watch live vehicle detection and tracking at the Crestwood intersection with advanced traffic camera AI.",
        "url": f"{BASE_URL}/",
        "image": f"{BASE_URL}/static/screenshot_afternoon.png",
    },
    "about": {
        "description": "Learn about Stop Sign Nanny's real-time computer vision pipeline for AI traffic monitoring. Built with YOLO detection, FastAPI, and advanced stop sign compliance tracking technology.",
        "url": f"{BASE_URL}/about",
        "image": f"{BASE_URL}/static/screenshot_afternoon.png",
    },
    "records": {
        "description": "View vehicle records and stop sign compliance statistics from our AI traffic monitoring system. Browse historical data from real-time vehicle detection and YOLO computer vision analysis.",
        "url": f"{BASE_URL}/records",
        "image": f"{BASE_URL}/static/screenshot_afternoon.png",
    },
    "vehicles": {
        "description": "Explore vehicle classification data from AI traffic monitoring. See breakdowns by type, color, and make/model from real-time YOLO detection and vision analysis.",
        "url": f"{BASE_URL}/vehicles",
        "image": f"{BASE_URL}/static/screenshot_afternoon.png",
    },
    "debug": {
        "description": "Debug interface for Stop Sign Nanny AI traffic monitoring system configuration and zone adjustment.",
        "url": f"{BASE_URL}/debug",
        "image": f"{BASE_URL}/static/screenshot_afternoon.png",
    },
}


def build_json_ld(base_url, metadata, page_type):
    """Build JSON-LD structured data for a page."""
    website_schema = {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "name": "Stop Sign Nanny",
        "url": base_url,
        "description": "Real-time AI traffic monitoring using YOLO computer vision for stop sign compliance detection at a neighborhood intersection",
    }

    creator_schema = {
        "@context": "https://schema.org",
        "@type": "Person",
        "name": "David Rose",
        "url": "https://github.com/cipher982",
        "sameAs": ["https://github.com/cipher982"],
    }

    web_app_schema = {
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": "Stop Sign Nanny",
        "description": metadata["description"],
        "url": metadata["url"],
        "applicationCategory": "ComputerVisionApplication",
        "operatingSystem": "Web",
        "image": metadata["image"],
        "author": {"@type": "Person", "name": "David Rose", "url": "https://github.com/cipher982"},
        "featureList": [
            "Real-time YOLO object detection",
            "Vehicle tracking and classification",
            "Stop sign compliance monitoring",
            "Live HLS video streaming",
            "Historical vehicle pass records",
        ],
        "softwareVersion": "2.0",
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
    }

    schemas = [website_schema, creator_schema, web_app_schema]

    if page_type == "home":
        video_schema = {
            "@context": "https://schema.org",
            "@type": "VideoObject",
            "name": "Live Stop Sign Traffic Monitoring",
            "description": "Real-time video stream showing AI-powered vehicle detection and stop sign compliance monitoring",
            "thumbnailUrl": f"{base_url}/static/screenshot_afternoon.png",
            "uploadDate": "2024-01-01T00:00:00Z",
            "contentUrl": f"{base_url}/stream/stream.m3u8",
            "embedUrl": f"{base_url}/",
        }
        schemas.append(video_schema)
    elif page_type == "records":
        dataset_schema = {
            "@context": "https://schema.org",
            "@type": "Dataset",
            "name": "Vehicle Pass Records",
            "description": "Historical dataset of vehicle detections including speed, stop duration, and compliance metrics from AI traffic monitoring",
            "url": f"{base_url}/records",
            "temporalCoverage": "2024-01/..",
            "variableMeasured": ["vehicle speed", "stop duration", "time in zone", "stop compliance"],
        }
        schemas.append(dataset_schema)

    return {"@context": "https://schema.org", "@graph": schemas}
