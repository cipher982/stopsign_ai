/**
 * Video Player JavaScript
 * Handles HLS video streaming and playback
 */

// Track if video player is already initialized
let videoPlayerInitialized = false;

// Video player initialization
function initializeVideoPlayer() {
    const video = document.getElementById('videoPlayer');
    if (!video) {
        console.log('Video player element not found, will try again later');
        return;
    }
    
    // Prevent multiple initializations, but allow retry if previous attempt failed
    if (videoPlayerInitialized && video.hlsInstance) {
        console.log('Video player already initialized, skipping');
        return;
    }
    
    // Clean up any existing instance before re-initializing
    if (video.hlsInstance) {
        console.log('Cleaning up existing HLS instance');
        video.hlsInstance.destroy();
        video.hlsInstance = null;
    }
    
    videoPlayerInitialized = true;
    const streamUrl = '/stream/stream.m3u8';
    
    console.log('Attempting to load video');
    
    if (typeof Hls !== 'undefined' && Hls.isSupported()) {
        console.log('HLS is supported');
        const hls = new Hls({debug: false});
        hls.loadSource(streamUrl);
        hls.attachMedia(video);
        
        hls.on(Hls.Events.MANIFEST_PARSED, function() {
            console.log('Manifest parsed, attempting to play');
            video.play().catch(e => console.log('Play failed:', e));
        });
        
        hls.on(Hls.Events.ERROR, function(event, data) {
            console.error('HLS error:', event, data);
            if (data.fatal) {
                switch(data.type) {
                    case Hls.ErrorTypes.NETWORK_ERROR:
                        console.log('Network error, trying to recover...');
                        hls.startLoad();
                        break;
                    case Hls.ErrorTypes.MEDIA_ERROR:
                        console.log('Media error, trying to recover...');
                        hls.recoverMediaError();
                        break;
                    default:
                        console.log('Fatal error, destroying HLS');
                        hls.destroy();
                        videoPlayerInitialized = false;
                        break;
                }
            }
        });
        
        // Store HLS instance for cleanup if needed
        video.hlsInstance = hls;
        
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        console.log('HLS not supported, but can play HLS natively');
        video.src = streamUrl;
        
        video.addEventListener('loadedmetadata', function() {
            console.log('Metadata loaded, attempting to play');
            video.play().catch(e => console.log('Play failed:', e));
        });
    } else {
        console.error('HLS is not supported and cannot play HLS natively');
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

// Initialize video when DOM is ready, with single retry
document.addEventListener('DOMContentLoaded', function() {
    // Try to initialize immediately
    initializeVideoPlayer();
    
    // Single retry after delay for HTMX loaded content
    setTimeout(function() {
        if (!videoPlayerInitialized) {
            initializeVideoPlayer();
        }
    }, 1000);
});

// Listen for HTMX events only if video isn't initialized yet
document.addEventListener('htmx:afterSettle', function() {
    if (!videoPlayerInitialized) {
        setTimeout(initializeVideoPlayer, 100);
    }
});