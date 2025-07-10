#!/usr/bin/env python3
"""
Clean up all StopSign dashboards from Grafana.
This script deletes dashboards via the Grafana API.
"""

import os
import sys

import requests


def get_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        print(f"Environment variable '{key}' is required")
        sys.exit(1)
    return value


def cleanup_dashboards(grafana_url: str, api_key: str):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Get all dashboards
    search_url = f"{grafana_url.rstrip('/')}/api/search?type=dash-db"
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        dashboards = response.json()
    except requests.RequestException as e:
        print(f"Failed to get dashboard list: {e}")
        return

    # Find StopSign dashboards
    stopsign_dashboards = [
        d for d in dashboards if any(keyword in d.get("title", "").lower() for keyword in ["stopsign", "stop sign"])
    ]

    if not stopsign_dashboards:
        print("No StopSign dashboards found to delete.")
        return

    print(f"Found {len(stopsign_dashboards)} StopSign dashboard(s) to delete:")
    for dash in stopsign_dashboards:
        print(f"  - {dash.get('title')} (UID: {dash.get('uid')})")

    # Delete each dashboard
    deleted_count = 0
    for dash in stopsign_dashboards:
        uid = dash.get("uid")
        title = dash.get("title", "Unknown")

        if not uid:
            print(f"  ✗ Cannot delete '{title}' - no UID")
            continue

        delete_url = f"{grafana_url.rstrip('/')}/api/dashboards/uid/{uid}"
        try:
            response = requests.delete(delete_url, headers=headers, timeout=10)
            if response.status_code == 200:
                print(f"  ✓ Deleted '{title}'")
                deleted_count += 1
            else:
                print(f"  ✗ Failed to delete '{title}' - HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"  ✗ Failed to delete '{title}' - {e}")

    print(f"\nDeleted {deleted_count} of {len(stopsign_dashboards)} dashboards.")


if __name__ == "__main__":
    grafana_url = get_env("GRAFANA_URL")
    api_key = get_env("GRAFANA_API_KEY")

    print("🧹 Cleaning up StopSign dashboards from Grafana...")
    cleanup_dashboards(grafana_url, api_key)
