"""Insight cache and rotating insights for the live stats panel."""

import logging
import random
from datetime import datetime
from datetime import timedelta

logger = logging.getLogger(__name__)


class InsightsCache:
    """Simple in-memory cache for insights with TTL expiration."""

    def __init__(self, ttl_seconds=45):
        self.ttl_seconds = ttl_seconds
        self._cache = {}

    def get(self, key):
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key, value):
        self._cache[key] = (value, datetime.now())

    def clear(self):
        self._cache.clear()


insights_cache = InsightsCache(ttl_seconds=45)


def get_real_insights(db, recent_passes):
    """Generate real insights based on actual data with caching."""
    cached_insights = insights_cache.get("insights")
    if cached_insights:
        return random.choice(cached_insights)

    insights = []

    try:
        peak_hour = db.get_peak_hour_today()
        if peak_hour:
            insights.append(f"Peak hour today: {peak_hour['display']} ({peak_hour['count']} vehicles)")

        avg_stop = db.get_average_stop_time(hours=24)
        if avg_stop and avg_stop["sample_size"] >= 5:
            insights.append(
                f"Average stop time: {avg_stop['avg_time_in_zone']}s (last {avg_stop['sample_size']} vehicles)"
            )

        fastest = db.get_fastest_vehicle_today()
        if fastest:
            time_str = fastest["time"].strftime("%I:%M %p") if fastest["time"] else "unknown time"
            insights.append(f"Fastest vehicle today: {fastest['speed']} px/s at {time_str}")

        streak = db.get_compliance_streak()
        if streak and streak["length"] >= 3:
            insights.append(f"Best compliance streak: {streak['length']} vehicles in a row")

        summary = db.get_traffic_summary_today()
        if summary:
            if summary["total_vehicles"] >= 10:
                insights.append(
                    f"Traffic today: {summary['total_vehicles']} vehicles across {summary['active_hours']} hours"
                )
            if summary["compliance_rate"] >= 80:
                insights.append(f"Excellent compliance: {summary['compliance_rate']}% today")
            elif summary["compliance_rate"] <= 50:
                insights.append(f"Low compliance: {summary['compliance_rate']}% today")

        if not insights:
            insights = [
                f"Fastest recent: {max((p.min_speed for p in recent_passes), default=0):.1f} px/s",
                f"Recent activity: {len(recent_passes)} vehicles tracked",
                "Monitoring active: Stop sign compliance tracking",
            ]

        insights_cache.set("insights", insights)

    except Exception as e:
        logger.error(f"Error generating real insights: {e}")
        insights = [
            f"Recent activity: {len(recent_passes)} vehicles",
            f"Last vehicle: {max((p.min_speed for p in recent_passes), default=0):.1f} px/s",
            "System monitoring active",
        ]

    return random.choice(insights) if insights else "Monitoring active"
