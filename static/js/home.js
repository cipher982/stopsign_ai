/**
 * Home Page JavaScript
 * Provides lightweight helpers for live stats updates.
 */

function updateLiveStats(data) {
    if (data.compliance) {
        document.getElementById('complianceRate').textContent = data.compliance;
    }
    if (data.violations) {
        document.getElementById('violationCount').textContent = data.violations;
    }
    if (data.vehicles) {
        document.getElementById('vehicleCount').textContent = data.vehicles;
    }
    if (data.lastDetection) {
        document.getElementById('lastDetection').textContent = data.lastDetection;
    }
    if (data.trend) {
        document.getElementById('trendArrow').textContent = data.trend;
    }
    if (data.insight) {
        document.getElementById('rotatingInsight').textContent = data.insight;
    }
}

document.addEventListener('htmx:afterRequest', function(event) {
    if (!event.detail || !event.detail.xhr) {
        return;
    }

    var requestUrl = event.detail.xhr.responseURL || '';
    if (!requestUrl.includes('/api/live-stats')) {
        return;
    }

    var response = event.detail.xhr.response;
    if (!response) {
        return;
    }

    var parser = new DOMParser();
    var doc = parser.parseFromString(response, 'text/html');

    var statsData = {
        compliance: doc.querySelector('[data-stat="compliance"]')?.textContent,
        violations: doc.querySelector('[data-stat="violations"]')?.textContent,
        vehicles: doc.querySelector('[data-stat="vehicles"]')?.textContent,
        lastDetection: doc.querySelector('[data-stat="lastDetection"]')?.textContent,
        trend: doc.querySelector('[data-stat="trend"]')?.textContent,
        insight: doc.querySelector('[data-stat="insight"]')?.textContent,
    };

    updateLiveStats(statsData);
});

// Reveal below-the-fold sections as they enter the viewport, nudging the user to scroll.
(function revealOnScroll() {
    var sections = document.querySelectorAll('.scroll-reveal');
    if (!('IntersectionObserver' in window)) {
        sections.forEach(function (el) { el.classList.add('inview'); });
        return;
    }
    var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                entry.target.classList.add('inview');
                io.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12 });
    sections.forEach(function (el) { io.observe(el); });
})();
