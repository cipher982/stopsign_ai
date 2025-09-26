
let adjustmentMode = false;
let clickedPoints = [];
let currentZoneType = 'stop-line';  // Current zone being edited
let debugZonesVisible = false;

// Zone configuration
const zoneConfig = {
    'stop-line': {
        name: 'Stop Zone',
        color: '#ff0000',
        bgColor: 'rgba(255, 0, 0, 0.2)',
        clicksRequired: 4,
        description: 'Click 4 corners to define the stop zone rectangle'
    },
    'pre-stop': {
        name: 'Pre-Stop Line',
        color: '#ffff00',
        bgColor: 'rgba(255, 255, 0, 0.2)',
        clicksRequired: 2,
        description: 'Click 2 points to define the pre-stop detection line'
    },
    'capture': {
        name: 'Capture Line',
        color: '#00ff00',
        bgColor: 'rgba(0, 255, 0, 0.2)',
        clicksRequired: 2,
        description: 'Click 2 points to define the photo capture line'
    }
};

function selectZoneType(zoneType) {
    currentZoneType = zoneType;
    clickedPoints = [];
    clearClickMarkers();

    // Update zone selector buttons
    const buttons = document.querySelectorAll('.zone-selector');
    buttons.forEach(btn => {
        btn.classList.remove('active');
    });

    const activeButton = document.getElementById('zone-' + zoneType);
    activeButton.classList.add('active');

    // Update instructions
    const instructions = document.getElementById('zone-instructions');
    instructions.innerText = zoneConfig[zoneType].description;

    // Update status
    const status = document.getElementById('status');
    status.innerText = `Ready to adjust ${zoneConfig[zoneType].name}`;

    // Reset adjustment mode button
    const adjustBtn = document.getElementById('adjustmentModeBtn');
    adjustBtn.innerText = `Adjust ${zoneConfig[zoneType].name}`;
}

function toggleDebugZones() {
    debugZonesVisible = !debugZonesVisible;

    fetch('/api/toggle-debug-zones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: debugZonesVisible })
    })
    .then(response => response.json())
    .then(data => {
        const button = document.getElementById('debugZonesBtn');
        button.innerText = debugZonesVisible ? 'Hide Debug Zones' : 'Show Debug Zones';

        const status = document.getElementById('status');
        status.innerText = debugZonesVisible ? 'Debug zones visible on video' : 'Debug zones hidden';
    })
    .catch(error => {
        console.error('Error toggling debug zones:', error);
    });
}

function toggleAdjustmentMode() {
    adjustmentMode = !adjustmentMode;
    clickedPoints = [];

    const video = document.getElementById('videoPlayer');
    const button = document.getElementById('adjustmentModeBtn');
    const status = document.getElementById('status');
    const config = zoneConfig[currentZoneType];

    if (adjustmentMode) {
        video.style.cursor = 'crosshair';
        video.style.outline = `3px solid ${config.color}`;
        button.innerText = 'Cancel Adjustment';
        status.innerText = `ADJUSTMENT MODE: ${config.description}`;
    } else {
        video.style.cursor = 'default';
        video.style.outline = 'none';
        button.innerText = `Adjust ${config.name}`;
        status.innerText = '';
        clearClickMarkers();
    }
}

function handleVideoClick(event) {
    if (!adjustmentMode) return;

    const config = zoneConfig[currentZoneType];

    // Prevent more clicks than required
    if (clickedPoints.length >= config.clicksRequired) return;

    const video = event.target;
    const rect = video.getBoundingClientRect();

    // Get click coordinates relative to video element
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    console.log('Click debug:', {
        zoneType: currentZoneType,
        browserClick: { x, y },
        videoElement: { width: rect.width, height: rect.height },
        actualVideo: { width: video.videoWidth, height: video.videoHeight }
    });

    clickedPoints.push({x: x, y: y});

    // Position marker at exact click location with zone color
    addClickMarker(rect.left + x, rect.top + y, clickedPoints.length, config.color);

    const status = document.getElementById('status');
    const submitBtn = document.getElementById('submitBtn');

    if (clickedPoints.length < config.clicksRequired) {
        const remaining = config.clicksRequired - clickedPoints.length;
        if (currentZoneType === 'stop-line') {
            status.innerText = `Point ${clickedPoints.length} of 4 set - ${remaining} more corners to go`;
        } else {
            status.innerText = `Point ${clickedPoints.length} of 2 set - click ${remaining} more point(s) for the line`;
        }
    }

    if (clickedPoints.length === config.clicksRequired) {
        status.innerText = `ALL POINTS SET ✓ - Click SUBMIT to update ${config.name}.`;
        submitBtn.style.display = 'inline-block';
        submitBtn.disabled = false;
    }
}

function updateZoneFromClicks() {
    const video = document.getElementById('videoPlayer');
    const config = zoneConfig[currentZoneType];

    const data = {
        zone_type: currentZoneType,
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

    const endpoint = currentZoneType === 'stop-line' ?
        '/api/update-stop-zone-from-display' :
        '/api/update-zone-from-display';

    fetch(endpoint, {
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
            // Display version and coordinates
            let statusText = `✅ SUCCESS! ${config.name} updated successfully.`;
            if (data.version) {
                statusText += ` (v${data.version})`;
            }

            if (currentZoneType === 'stop-line' && Array.isArray(data.stop_zone)) {
                const formatted = data.stop_zone
                    .map((point, index) => `Corner ${index + 1}: (${Math.round(point[0])}, ${Math.round(point[1])})`)
                    .join('\n');
                statusText += `\n${formatted}`;
            } else if (data.zone_data) {
                const linePoints = data.zone_data.pre_stop_line || data.zone_data.capture_line;
                if (Array.isArray(linePoints) && linePoints.length === 2) {
                    const [p1, p2] = linePoints;
                    statusText += `\nLine: (${Math.round(p1[0])}, ${Math.round(p1[1])}) → (${Math.round(p2[0])}, ${Math.round(p2[1])})`;
                }
            }

            status.innerText = statusText;

            // Show coordinate details in console
            console.log('Config update response:', data);

            setTimeout(() => {
                status.innerText = 'Ready for next adjustment.';
                toggleAdjustmentMode();
            }, 3000);
        } else {
            status.innerText = '❌ ERROR: ' + (data.error || 'Unknown error occurred');
        }
    })
    .catch((error) => {
        console.error('Error:', error);
        const status = document.getElementById('status');
        const zoneName = zoneConfig[currentZoneType].name;
        status.innerText = `❌ NETWORK ERROR: Could not update ${zoneName}.`;
    });
}

function resetPoints() {
    clickedPoints = [];
    clearClickMarkers();
    const status = document.getElementById('status');
    const submitBtn = document.getElementById('submitBtn');
    const zoneName = zoneConfig[currentZoneType].name;
    const clicksRequired = zoneConfig[currentZoneType].clicksRequired;
    status.innerText = `Points cleared. Click ${clicksRequired} new point(s) to update the ${zoneName}.`;
    submitBtn.style.display = 'none';
    submitBtn.disabled = true;
}

function addClickMarker(pageX, pageY, pointNumber, color = '#ff0000') {
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
    marker.style.backgroundColor = color;
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

function clearClickMarkers() {
    const markers = document.querySelectorAll('.click-marker');
    markers.forEach(marker => marker.remove());
}

function debugCoordinates() {
    const video = document.getElementById('videoPlayer');
    if (!video) return;

    const data = {
        display_points: [{x: 100, y: 100}, {x: 500, y: 300}],
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
        document.getElementById('debugOutput').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
    });
}

function showCoordinateInfo() {
    fetch('/api/coordinate-info')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('coordOutput').innerHTML = '<div style="color: #ff6b6b; padding: 10px; background: #2a1f1f; border-radius: 5px;">Error: ' + data.error + '<br><br>This usually means the video analyzer is not running yet. Try starting the video processing service first.</div>';
            } else {
                document.getElementById('coordOutput').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
            }
        })
        .catch(error => {
            document.getElementById('coordOutput').innerHTML = '<div style="color: #ff6b6b;">Network error: ' + error.message + '</div>';
        });
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the interface
    selectZoneType('stop-line'); // Set default zone

    setTimeout(() => {
        const video = document.getElementById('videoPlayer');
        if (video) {
            video.addEventListener('click', handleVideoClick);
        }
    }, 1000);
});
