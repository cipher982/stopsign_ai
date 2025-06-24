# Sample Data for Local Development

This directory contains sample video files for local development and testing.

## Files

- `sample.mp4` - A small sample video file for testing the local development stack
  - Should be ≤ 5MB in size for quick downloads
  - Contains vehicles or objects that can be detected by YOLO models
  - Loops continuously for testing

## Usage

The local development environment is configured to use `file:///app/sample_data/sample.mp4` as the default RTSP source when `ENV=local`.

To provide your own sample video:
1. Place a `.mp4` file in this directory
2. Update the `RTSP_URL` in your `.env` file to point to your video:
   ```
   RTSP_URL=file:///app/sample_data/your_video.mp4
   ```

## Creating Sample Videos

If you need to create a sample video file, you can:

1. Use ffmpeg to create a test pattern:
   ```bash
   ffmpeg -f lavfi -i testsrc=duration=10:size=640x480:rate=15 -c:v libx264 sample.mp4
   ```

2. Or download a Creative Commons licensed video file suitable for testing

3. Convert existing video to smaller size:
   ```bash
   ffmpeg -i input.mp4 -vf scale=640:480 -t 30 -c:v libx264 -crf 28 sample.mp4
   ```