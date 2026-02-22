/**
 * Video Player JavaScript
 * Handles HLS video streaming, playback, and stream status display.
 */

// Track initialization state to prevent multiple instances
var videoPlayerInitialized = false;

function setStreamStatus(state) {
    var statusEl = document.getElementById('streamStatus');
    if (!statusEl) return;
    var dot = statusEl.querySelector('.stream-status-dot');
    var label = statusEl.querySelector('.stream-status-label');
    if (!dot || !label) return;

    dot.className = 'stream-status-dot';
    switch (state) {
        case 'live':
            dot.classList.add('live-pulse');
            dot.style.background = 'var(--ok)';
            label.textContent = 'Live';
            break;
        case 'connecting':
            dot.classList.add('live-pulse');
            dot.style.background = 'var(--warn)';
            label.textContent = 'Connecting...';
            break;
        case 'offline':
            dot.style.background = 'var(--bad)';
            label.textContent = 'Offline';
            break;
    }
}

function showStreamError(message) {
    var container = document.getElementById('videoContainer');
    if (!container) return;
    var existing = container.querySelector('.stream-error');
    if (existing) existing.remove();

    var el = document.createElement('div');
    el.className = 'stream-error';
    el.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:24px;">' +
        '<div style="font-size:18px;margin-bottom:8px;">&#128247;</div>' +
        '<div style="font-size:14px;font-weight:600;margin-bottom:4px;">Stream Unavailable</div>' +
        '<div style="font-size:13px;">' + message + '</div>' +
        '<div style="font-size:12px;margin-top:8px;color:var(--text-secondary);">Retrying automatically...</div>' +
        '</div>';
    el.style.position = 'absolute';
    el.style.inset = '0';
    el.style.display = 'flex';
    el.style.alignItems = 'center';
    el.style.justifyContent = 'center';
    el.style.background = '#000';
    el.style.zIndex = '5';
    container.appendChild(el);
}

function clearStreamError() {
    var container = document.getElementById('videoContainer');
    if (!container) return;
    var existing = container.querySelector('.stream-error');
    if (existing) existing.remove();
}

// Video player initialization
function initializeVideoPlayer() {
    var video = document.getElementById('videoPlayer');
    if (!video) {
        console.log('Video player element not found, will try again later');
        return;
    }

    // Prevent multiple initializations
    if (videoPlayerInitialized || video.hlsInstance) {
        console.log('Video player already initialized, skipping');
        return;
    }

    var streamUrl = '/stream/stream.m3u8';
    console.log('Attempting to load video');
    setStreamStatus('connecting');

    if (typeof Hls !== 'undefined' && Hls.isSupported()) {
        console.log('HLS is supported');
        var hls = new Hls({
            debug: false,
            liveSyncDurationCount: 5,        // 10s target buffer — absorbs pipeline variance
            liveMaxLatencyDurationCount: 10,
            maxLiveSyncPlaybackRate: 1.5,    // faster catch-up vs hard jump
            liveDurationInfinity: true
        });

        hls.on(Hls.Events.MANIFEST_PARSED, function() {
            console.log('Manifest parsed, attempting to play');
            clearStreamError();
            setStreamStatus('live');
            video.play().catch(function(e) { console.log('Play failed:', e); });
        });

        hls.on(Hls.Events.ERROR, function(event, data) {
            console.error('HLS error:', event, data);
            if (data.fatal) {
                console.log('Fatal HLS error, attempting recovery');
                switch(data.type) {
                case Hls.ErrorTypes.NETWORK_ERROR:
                    console.log('Network error - stream may be unavailable');
                    setStreamStatus('offline');
                    showStreamError('The camera feed is not reachable right now. This can happen during maintenance or network issues.');
                    break;
                case Hls.ErrorTypes.MEDIA_ERROR:
                    console.log('Media error - trying to recover');
                    hls.recoverMediaError();
                    return;
                default:
                    console.log('Unrecoverable error');
                    setStreamStatus('offline');
                    showStreamError('An unexpected error occurred with the video stream.');
                    break;
                }
                hls.destroy();
                videoPlayerInitialized = false;
                video.hlsInstance = null;
                // Retry after delay
                setTimeout(initializeVideoPlayer, 5000);
            }
        });

        hls.on(Hls.Events.MANIFEST_LOADING, function() {
            console.log('Loading HLS manifest...');
        });

        // Store HLS instance for cleanup and prevent re-initialization
        video.hlsInstance = hls;
        videoPlayerInitialized = true;

        hls.loadSource(streamUrl);
        hls.attachMedia(video);

    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        console.log('HLS not supported, but can play HLS natively');
        video.src = streamUrl;
        videoPlayerInitialized = true;

        video.addEventListener('loadedmetadata', function() {
            console.log('Metadata loaded, attempting to play');
            clearStreamError();
            setStreamStatus('live');
            video.play().catch(function(e) { console.log('Play failed:', e); });
        });
    } else {
        console.error('HLS is not supported and cannot play HLS natively');
        setStreamStatus('offline');
        showStreamError('Your browser does not support HLS video playback.');
    }

    // Add error listener
    video.addEventListener('error', function(e) {
        console.error('Video error:', e);
    });

    // Log video dimensions when metadata is loaded
    video.addEventListener('loadedmetadata', function() {
        console.log('Video dimensions:', {
            videoWidth: video.videoWidth,
            videoHeight: video.videoHeight,
            displayWidth: video.clientWidth,
            displayHeight: video.clientHeight
        });
    });
}

// Initialize video when DOM is ready, with retry logic for HTMX loaded content
document.addEventListener('DOMContentLoaded', function() {
    initializeVideoPlayer();
    setTimeout(initializeVideoPlayer, 1000);
});

// Also listen for HTMX events in case video is loaded dynamically
document.addEventListener('htmx:afterSettle', function(event) {
    if (event.detail.target.id === 'videoContainer' ||
        event.detail.target.querySelector('#videoPlayer')) {
        setTimeout(initializeVideoPlayer, 200);
    }
});
