/**
 * Video Player JavaScript
 * Handles HLS video streaming and playback
 */

// Track initialization state to prevent multiple instances
let videoPlayerInitialized = false;

// Video player initialization
function initializeVideoPlayer() {
    const video = document.getElementById('videoPlayer');
    if (!video) {
        console.log('Video player element not found, will try again later');
        return;
    }
    
    // Prevent multiple initializations
    if (videoPlayerInitialized || video.hlsInstance) {
        console.log('Video player already initialized, skipping');
        return;
    }
    
    const streamUrl = '/stream/stream.m3u8';
    console.log('Attempting to load video');
    
    if (typeof Hls !== 'undefined' && Hls.isSupported()) {
        console.log('HLS is supported');
        const hls = new Hls({
            debug: false,
            lowLatencyMode: true,
            liveSyncDurationCount: 1,        
            liveMaxLatencyDurationCount: 3,  
            maxLiveSyncPlaybackRate: 1.5,    
            liveDurationInfinity: true,      
            enableWorker: true               
        });
        
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
        
        // Add manifest loading timeout
        hls.on(Hls.Events.MANIFEST_LOADING, function() {
            console.log('Loading HLS manifest...');
        });
        
        // Force immediate jump to live edge for live streams
        hls.on(Hls.Events.LEVEL_UPDATED, function(event, data) {
            if (data.details.live && video.currentTime < hls.liveSyncPosition - 2) {
                console.log('Jumping to live edge:', hls.liveSyncPosition);
                video.currentTime = hls.liveSyncPosition - 0.5;
            }
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

// Initialize video when DOM is ready, with retry logic for HTMX loaded content
document.addEventListener('DOMContentLoaded', function() {
    // Try to initialize immediately
    initializeVideoPlayer();
    
    // Also try after a delay in case video is loaded via HTMX
    setTimeout(initializeVideoPlayer, 1000);
});

// Also listen for HTMX events in case video is loaded dynamically
document.addEventListener('htmx:afterSettle', function(event) {
    // Only initialize if the video container was updated
    if (event.detail.target.id === 'videoContainer' || 
        event.detail.target.querySelector('#videoPlayer')) {
        setTimeout(initializeVideoPlayer, 200);
    }
});