/**
 * Home Page JavaScript
 * Handles stop line adjustment and recent passes functionality
 */

// Stop zone adjustment state
let adjustmentMode = false;
let clickedPoints = [];
let coordinateInfo = null;
let currentZoneType = 'stop-line';

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

// Load coordinate system information
function loadCoordinateInfo() {
    fetch('/api/coordinate-info')
        .then(response => response.json())
        .then(data => {
            coordinateInfo = data;
            console.log('Coordinate info loaded:', data);
        })
        .catch(error => {
            console.error('Error loading coordinate info:', error);
        });
}

// Toggle click-to-set adjustment mode
function toggleAdjustmentMode() {
    adjustmentMode = !adjustmentMode;
    clickedPoints = [];
    
    const video = document.getElementById('videoPlayer');
    const button = document.getElementById('adjustmentModeBtn');
    const status = document.getElementById('status');
    
    if (adjustmentMode) {
        video.style.cursor = 'crosshair';
        video.style.outline = '3px solid #ff0000';
        button.innerText = 'Cancel Adjustment';
        status.innerText = 'ADJUSTMENT MODE: Click two points on the video to set the stop line';
        loadCoordinateInfo();
    } else {
        video.style.cursor = 'default';
        video.style.outline = 'none';
        button.innerText = 'Adjust Stop Line';
        status.innerText = '';
        clearClickMarkers();
    }
}

// Handle video clicks for stop line adjustment
function handleVideoClick(event) {
    if (!adjustmentMode) return;
    
    const video = event.target;
    const rect = video.getBoundingClientRect();
    
    // Get click coordinates relative to video element
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Add click point
    clickedPoints.push({x: x, y: y});
    
    // Add visual marker
    addClickMarker(event.clientX, event.clientY, clickedPoints.length);
    
    const status = document.getElementById('status');
    
    if (clickedPoints.length === 1) {
        status.innerText = 'Good! Now click the second point for the stop line.';
    } else if (clickedPoints.length === 2) {
        // Send both points to update the stop zone
        updateStopZoneFromClicks();
    }
}

// Update stop zone using clicked coordinates
function updateStopZoneFromClicks() {
    const video = document.getElementById('videoPlayer');
    
    const data = {
        display_points: clickedPoints,
        video_element_size: {
            width: video.clientWidth,
            height: video.clientHeight
        },
        actual_video_size: {
            width: video.videoWidth,
            height: video.videoHeight
        }
    };
    
    fetch('/api/update-stop-zone-from-display', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        const status = document.getElementById('status');
        if (data.status === 'success') {
            status.innerText = 'Stop line updated successfully! Coordinates transformed from display to processing system.';
            console.log('Coordinate transformation details:', data);
            
            // Exit adjustment mode
            setTimeout(() => {
                toggleAdjustmentMode();
            }, 2000);
        } else {
            status.innerText = 'Error: ' + (data.error || 'Unknown error occurred');
            console.error('Update error:', data);
        }
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById('status').innerText = 'Network error occurred.';
    });
}

// Add visual click markers
function addClickMarker(pageX, pageY, pointNumber) {
    const video = document.getElementById('videoPlayer');
    const rect = video.getBoundingClientRect();
    
    // Ensure video has a wrapper for positioning
    let wrapper = video.parentElement;
    if (!wrapper || !wrapper.classList.contains('video-wrapper')) {
        wrapper = document.createElement('div');
        wrapper.className = 'video-wrapper';
        wrapper.style.position = 'relative';
        wrapper.style.display = 'inline-block';
        video.parentNode.insertBefore(wrapper, video);
        wrapper.appendChild(video);
    }
    
    // Calculate position relative to video element
    const relativeX = pageX - rect.left;
    const relativeY = pageY - rect.top;
    
    const marker = document.createElement('div');
    marker.className = 'click-marker';
    marker.innerHTML = pointNumber;
    marker.style.position = 'absolute';
    marker.style.left = (relativeX - 15) + 'px';
    marker.style.top = (relativeY - 15) + 'px';
    marker.style.width = '30px';
    marker.style.height = '30px';
    marker.style.backgroundColor = '#ff0000';
    marker.style.color = 'white';
    marker.style.borderRadius = '50%';
    marker.style.display = 'flex';
    marker.style.alignItems = 'center';
    marker.style.justifyContent = 'center';
    marker.style.fontWeight = 'bold';
    marker.style.fontSize = '14px';
    marker.style.zIndex = '9999';
    marker.style.pointerEvents = 'none';
    wrapper.appendChild(marker);
}

// Clear all click markers
function clearClickMarkers() {
    const markers = document.querySelectorAll('.click-marker');
    markers.forEach(marker => marker.remove());
}

// Track existing passes to avoid reloading unchanged images
let existingPasses = new Set();
let imageCache = new Map(); // Cache for preloaded images

// Fetch recent vehicle passes with smart updates
function fetchRecentPasses() {
    fetch('/api/recent-vehicle-passes')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then(html => {
            // Parse the new HTML to compare with existing
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const newPasses = doc.querySelectorAll('[id^="pass-"]');
            const currentContainer = document.getElementById('recentPasses');
            
            // Only update if we have significant changes
            const newPassIds = new Set(Array.from(newPasses).map(p => p.id));
            
            // If this is first load or we have new passes, update
            if (existingPasses.size === 0 || !setsEqual(existingPasses, newPassIds)) {
                // Preload new images before updating DOM
                preloadNewImages(doc).then(() => {
                    currentContainer.innerHTML = html;
                    existingPasses = newPassIds;
                    console.log('Updated recent passes - new content detected');
                });
            } else {
                console.log('No new passes, skipping update to preserve images');
            }
        })
        .catch(error => {
            console.error('Error fetching recent passes:', error);
            document.getElementById('recentPasses').innerHTML = `<p>Error fetching data: ${error.message}</p>`;
        });
}

// Helper function to compare sets
function setsEqual(set1, set2) {
    return set1.size === set2.size && [...set1].every(x => set2.has(x));
}

// Preload images to browser cache
function preloadNewImages(doc) {
    return new Promise((resolve) => {
        const images = doc.querySelectorAll('img[src]');
        const imagePromises = [];
        
        images.forEach(img => {
            const src = img.getAttribute('src');
            if (src && !imageCache.has(src)) {
                const imagePromise = new Promise((imgResolve) => {
                    const preloadImg = new Image();
                    preloadImg.onload = () => {
                        imageCache.set(src, true);
                        console.log('Preloaded image:', src);
                        imgResolve();
                    };
                    preloadImg.onerror = () => {
                        console.warn('Failed to preload image:', src);
                        imgResolve(); // Continue even if image fails
                    };
                    preloadImg.src = src;
                });
                imagePromises.push(imagePromise);
            }
        });
        
        if (imagePromises.length === 0) {
            resolve(); // No new images to preload
        } else {
            Promise.all(imagePromises).then(resolve);
        }
    });
}

// Debug mode for coordinate testing
function debugCoordinateTransform(testPoints) {
    const video = document.getElementById('videoPlayer');
    if (!video) {
        console.error('Video element not found');
        return;
    }
    
    const data = {
        display_points: testPoints || [
            {x: 100, y: 100},
            {x: 500, y: 300}
        ],
        video_element_size: {
            width: video.clientWidth,
            height: video.clientHeight
        }
    };
    
    fetch('/api/debug-coordinates', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        console.log('=== COORDINATE TRANSFORMATION DEBUG ===');
        console.log('Coordinate System Info:', data.coordinate_system_info);
        console.log('Transformation Chain:', data.transformation_debug);
        console.log('Point Transformations:', data.point_transformations);
        console.log('Current Stop Line (Display Coords):', data.current_stop_line_display);
        
        // Show roundtrip errors
        data.point_transformations.forEach(pt => {
            const error = Math.sqrt(pt.roundtrip_error.x ** 2 + pt.roundtrip_error.y ** 2);
            console.log(`Point ${pt.point_index + 1} roundtrip error: ${error.toFixed(2)}px`);
        });
    })
    .catch(error => {
        console.error('Debug error:', error);
    });
}

// Update live stats from API response
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

// Handle HTMX after request event for stats
document.addEventListener('htmx:afterRequest', function(event) {
    if (event.detail.xhr.responseURL.includes('/api/live-stats')) {
        const response = event.detail.xhr.response;
        if (response) {
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
        }
    }
});

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Fetch recent passes initially and set up smart refresh
    fetchRecentPasses();
    setInterval(fetchRecentPasses, 60000); // Check every 60 seconds, but only update if changed
    
    // Initialize video event listeners
    setTimeout(() => {
        const video = document.getElementById('videoPlayer');
        if (video) {
            video.addEventListener('click', handleVideoClick);
            console.log('Video click handler attached');
        }
        loadCoordinateInfo();
    }, 1000);
});