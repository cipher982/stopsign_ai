import asyncio
import json
from types import SimpleNamespace

import pytest

from stopsign.web.app import app
from stopsign.web.routes import health


def test_readyz_route_supports_get_and_head():
    matching_routes = [route for route in app.routes if getattr(route, "path", None) == "/readyz"]

    assert len(matching_routes) == 1
    assert matching_routes[0].methods == {"GET", "HEAD"}


def test_readyz_returns_monitor_payload(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(health, "_check_database", lambda db, query: None)
    monkeypatch.setattr(app.state, "db", object(), raising=False)
    request = SimpleNamespace(app=app)
    route = next(route for route in app.routes if getattr(route, "path", None) == "/readyz")
    response = asyncio.run(route.endpoint(request))
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert payload["schema_version"] == 2
    assert payload["ready"] is True
    assert payload["status"] == "healthy"
