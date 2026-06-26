import asyncio

from stopsign.web.app import app


def test_readyz_route_supports_get_and_head():
    matching_routes = [route for route in app.routes if getattr(route, "path", None) == "/readyz"]

    assert len(matching_routes) == 1
    assert matching_routes[0].methods == {"GET", "HEAD"}


def test_readyz_returns_monitor_payload():
    route = next(route for route in app.routes if getattr(route, "path", None) == "/readyz")
    response = asyncio.run(route.endpoint())

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.body == b'{"status":"ready"}'
