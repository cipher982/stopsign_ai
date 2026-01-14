/**
 * Video Player JavaScript
 * Handles video streaming from multiple sources:
 * - legacy_hls: Original ffmpeg-generated HLS (/stream/stream.m3u8)
 * - mediamtx_hls: MediaMTX Low-Latency HLS (configurable URL)
 * - mediamtx_webrtc: MediaMTX WebRTC/WHEP (lowest latency)
 */

// Track initialization state to prevent multiple instances
let videoPlayerInitialized = false;

// Get video config from page (injected by server)
function getVideoConfig() {
    const configElement = document.getElementById('video-config');
    if (configElement) {
        try {
            return JSON.parse(configElement.textContent);
        } catch (e) {
            console.error('Failed to parse video config:', e);
        }
    }
    // Default config for backwards compatibility
    return {
        source: 'legacy_hls',
        hlsUrl: '/stream/stream.m3u8',
        webrtcUrl: null
    };
}

// HLS player initialization (shared by legacy and mediamtx_hls)
function initializeHlsPlayer(video, streamUrl, isLowLatency = false) {
    if (typeof Hls !== 'undefined' && Hls.isSupported()) {
        console.log(`Initializing HLS player (low-latency: ${isLowLatency})`);

        // Config tuned for low-latency when using MediaMTX
        const hlsConfig = isLowLatency ? {
            debug: false,
            liveSyncDurationCount: 1,         // Stay close to live edge
            liveMaxLatencyDurationCount: 3,   // Allow 3 segments max latency
            maxLiveSyncPlaybackRate: 1.5,     // Catch up faster
            liveDurationInfinity: true,
            lowLatencyMode: true,             // Enable LL-HLS features
            backBufferLength: 10              // Keep less back buffer
        } : {
            debug: false,
            liveSyncDurationCount: 3,
            liveMaxLatencyDurationCount: 10,
            maxLiveSyncPlaybackRate: 1.2,
            liveDurationInfinity: true
        };

        const hls = new Hls(hlsConfig);

        hls.on(Hls.Events.MANIFEST_PARSED, function() {
            console.log('Manifest parsed, attempting to play');
            video.play().catch(e => console.log('Play failed:', e));
        });

        hls.on(Hls.Events.ERROR, function(event, data) {
            console.error('HLS error:', event, data);
            if (data.fatal) {
                console.log('Fatal HLS error, attempting recovery');
                switch(data.type) {
                case Hls.ErrorTypes.NETWORK_ERROR:
                    console.log('Network error - stream may be unavailable');
                    break;
                case Hls.ErrorTypes.MEDIA_ERROR:
                    console.log('Media error - trying to recover');
                    hls.recoverMediaError();
                    return;
                default:
                    console.log('Unrecoverable error');
                    break;
                }
                hls.destroy();
                videoPlayerInitialized = false;
                video.hlsInstance = null;
                // Retry after delay
                setTimeout(initializeVideoPlayer, 3000);
            }
        });

        hls.on(Hls.Events.MANIFEST_LOADING, function() {
            console.log('Loading HLS manifest from:', streamUrl);
        });

        video.hlsInstance = hls;
        videoPlayerInitialized = true;

        hls.loadSource(streamUrl);
        hls.attachMedia(video);

    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        // Safari native HLS
        console.log('Using native HLS playback');
        video.src = streamUrl;
        videoPlayerInitialized = true;

        video.addEventListener('loadedmetadata', function() {
            console.log('Metadata loaded, attempting to play');
            video.play().catch(e => console.log('Play failed:', e));
        });
    } else {
        console.error('HLS is not supported');
    }
}

// WebRTC/WHEP player initialization
async function initializeWebRTCPlayer(video, whepUrl) {
    console.log('Initializing WebRTC player with WHEP URL:', whepUrl);

    try {
        const pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });

        // Handle incoming tracks
        pc.ontrack = (event) => {
            console.log('Received track:', event.track.kind);
            if (event.streams && event.streams[0]) {
                video.srcObject = event.streams[0];
                video.play().catch(e => console.log('Play failed:', e));
            }
        };

        pc.oniceconnectionstatechange = () => {
            console.log('ICE connection state:', pc.iceConnectionState);
            if (pc.iceConnectionState === 'failed' || pc.iceConnectionState === 'disconnected') {
                console.log('WebRTC connection failed, will retry...');
                pc.close();
                videoPlayerInitialized = false;
                video.webrtcPc = null;
                setTimeout(initializeVideoPlayer, 3000);
            }
        };

        // Add transceivers for receiving video (and optionally audio)
        pc.addTransceiver('video', { direction: 'recvonly' });

        // Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // Wait for ICE gathering to complete (or timeout)
        await new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
                // Timeout after 2 seconds
                setTimeout(resolve, 2000);
            }
        });

        // Send offer to WHEP endpoint
        const response = await fetch(whepUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/sdp' },
            body: pc.localDescription.sdp
        });

        if (!response.ok) {
            throw new Error(`WHEP request failed: ${response.status}`);
        }

        const answerSdp = await response.text();
        await pc.setRemoteDescription({
            type: 'answer',
            sdp: answerSdp
        });

        video.webrtcPc = pc;
        videoPlayerInitialized = true;
        console.log('WebRTC connection established');

    } catch (error) {
        console.error('WebRTC initialization failed:', error);
        videoPlayerInitialized = false;
        // Retry after delay
        setTimeout(initializeVideoPlayer, 3000);
    }
}

// Main initialization function
function initializeVideoPlayer() {
    const video = document.getElementById('videoPlayer');
    if (!video) {
        console.log('Video player element not found, will try again later');
        return;
    }

    // Check if THIS specific element is already initialized (handles HTMX element replacement)
    if (video.hlsInstance || video.webrtcPc) {
        console.log('Video player already initialized, skipping');
        return;
    }

    // Reset global flag if element was replaced by HTMX
    videoPlayerInitialized = false;

    const config = getVideoConfig();
    console.log('Video config:', config);

    switch (config.source) {
        case 'mediamtx_webrtc':
            if (config.webrtcUrl) {
                initializeWebRTCPlayer(video, config.webrtcUrl);
            } else {
                console.error('WebRTC URL not configured, falling back to HLS');
                initializeHlsPlayer(video, config.hlsUrl, true);
            }
            break;

        case 'mediamtx_hls':
            initializeHlsPlayer(video, config.hlsUrl, true);
            break;

        case 'legacy_hls':
        default:
            initializeHlsPlayer(video, '/stream/stream.m3u8', false);
            break;
    }

    // Common event listeners
    video.addEventListener('error', function(e) {
        console.error('Video error:', e);
    });

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
