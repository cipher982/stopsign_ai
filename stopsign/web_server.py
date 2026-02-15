"""Entry point shim — preserves `python -m stopsign.web_server` and Docker CMD compatibility.

All logic lives in stopsign.web.app; this module re-exports `app` for uvicorn
and delegates `main()` so existing scripts keep working.
"""

import logging

from stopsign.web.app import app  # noqa: F401
from stopsign.web.app import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
