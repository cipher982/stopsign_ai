/**
 * Home Page JavaScript
 * Provides lightweight helpers for live stats updates.
 */

function updateLiveStats(data) {
    if (data.complianceData) {
        document.getElementById('complianceRate').textContent = data.complianceData;
    }
    if (data.violationData) {
        document.getElementById('violationCount').textContent = data.violationData;
    }
    if (data.vehicleData) {
        document.getElementById('vehicleCount').textContent = data.vehicleData;
    }
    if (data.lastDetectionData) {
        document.getElementById('lastDetection').textContent = data.lastDetectionData;
    }
    if (data.trendData) {
        document.getElementById('trendArrow').textContent = data.trendData;
    }
    if (data.insightData) {
        document.getElementById('rotatingInsight').textContent = data.insightData;
    }
}

document.addEventListener('htmx:afterRequest', function(event) {
    if (!event.detail || !event.detail.xhr) {
        return;
    }

    const requestUrl = event.detail.xhr.responseURL || '';
    if (!requestUrl.includes('/api/live-stats')) {
        return;
    }

    const response = event.detail.xhr.response;
    if (!response) {
        return;
    }

    const parser = new DOMParser();
    const doc = parser.parseFromString(response, 'text/html');

    const statsData = {
        complianceData: doc.getElementById('complianceData')?.textContent,
        violationData: doc.getElementById('violationData')?.textContent,
        vehicleData: doc.getElementById('vehicleData')?.textContent,
        lastDetectionData: doc.getElementById('lastDetectionData')?.textContent,
        trendData: doc.getElementById('trendData')?.textContent,
        insightData: doc.getElementById('insightData')?.textContent,
    };

    updateLiveStats(statsData);
});
