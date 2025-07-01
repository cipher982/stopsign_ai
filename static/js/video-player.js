/**
 * Video Player JavaScript
 * Handles HLS video streaming and playback
 */

// Video player initialization
function initializeVideoPlayer() {
    const video = document.getElementById('videoPlayer');
    if (!video) {
        console.log('Video player element not found, will try again later');
        return;
    }
    const streamUrl = '/stream/stream.m3u8';
    
    console.log('Attempting to load video');
    
    if (typeof Hls !== 'undefined' && Hls.isSupported()) {
        console.log('HLS is supported');
        const hls = new Hls({debug: false});
        hls.loadSource(streamUrl);
        hls.attachMedia(video);
        
        hls.on(Hls.Events.MANIFEST_PARSED, function() {
            console.log('Manifest parsed, attempting to play');
            video.play();
        });
        
        hls.on(Hls.Events.ERROR, function(event, data) {
            console.error('HLS error:', event, data);
        });
        
        // Store HLS instance for cleanup if needed
        video.hlsInstance = hls;
        
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        console.log('HLS not supported, but can play HLS natively');
        video.src = streamUrl;
        
        video.addEventListener('loadedmetadata', function() {
            console.log('Metadata loaded, attempting to play');
            video.play();
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
    setTimeout(initializeVideoPlayer, 500);
    setTimeout(initializeVideoPlayer, 1500);
});

// Also listen for HTMX events in case video is loaded dynamically
document.addEventListener('htmx:afterSettle', function() {
    setTimeout(initializeVideoPlayer, 100);
});