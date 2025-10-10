# Stop Sign AI - Deployment & Architecture

## Deployment Overview
This service is deployed to **cube** via Coolify and runs in multiple Docker containers:
1. **rtsp_to_redis** - Standalone service that captures RTSP stream and pushes frames to Redis
2. **Main compose stack** (docker/production/docker-compose.yml):
   - `video_analyzer` - AI detection and tracking (GPU-accelerated)
   - `ffmpeg_service` - HLS stream generation for web viewing
   - `web_server` - FastAPI web interface and API

## Data Persistence Architecture

### PostgreSQL Database (clifford, NOT cube)
- **Location**: `5.161.97.53:5432/stopsign`
- **Container**: `kgcos0o4cw4ok0ss0g08wswo` on clifford server
- **Tables**:
  - `vehicle_passes` - Core metrics (timestamp, speed, stop duration, time_in_zone, image_path)
  - `car_state_history` - Full tracking history per vehicle
  - `config_settings` - Dynamic configuration with version history
- **Current Data**: ~41k passes spanning Oct 2024 - Oct 2025 (~1 year retention)
- **Environment Var**: `DB_URL` in cube containers points to clifford's PostgreSQL

### MinIO S3 Storage (clifford, NOT cube)
- **Endpoint**: `minio-nwcs0c4g0w8gcgow0gscgckg.5.161.97.53.sslip.io`
- **Public URL**: `https://api.files.drose.io`
- **Container**: `minio-agog4sg0488g0080woo00ks4` on clifford
- **Bucket**: `vehicle-images`
- **Storage**: ~1.5 GiB, 119k+ images
- **Format**: Cropped vehicle JPEGs named `vehicle_{uuid}_{timestamp}.jpg`
- **Image Path in DB**: Stored as `minio://vehicle-images/vehicle_*.jpg`

### Data Flow
1. **RTSP Camera** → rtsp_to_redis container → **Redis** (raw frames on cube)
2. **video_analyzer** reads frames from Redis → YOLO detection → tracking logic
3. When vehicle exits stop zone:
   - Cropped vehicle image → **MinIO** (stopsign/tracking.py:547-593)
   - Metrics record → **PostgreSQL** (stopsign/database.py:153-169)
   - Both on **clifford** via network calls from cube
4. **web_server** queries PostgreSQL + serves images via MinIO public URL

## Key Implementation Details

### Image Capture
- Images captured at "capture line" crossing (configurable via debug UI)
- 10% padding added around vehicle bounding box
- Uploaded synchronously to MinIO during vehicle pass processing
- Path stored in database as `image_path` field for 1:1 mapping

### Configuration Management
- Config file: `/app/config/config.yaml` (mounted volume in containers)
- Dynamic updates via web UI stored in `config_settings` table with full history
- Stop zones, lines, thresholds all configurable and versioned

### Redis Usage
- Ephemeral frame buffer only (no persistence)
- Keys: `raw_frames` (RTSP input), `processed_frames` (annotated output)
- Used for inter-container communication on cube

## Troubleshooting Data Issues
- Database queries: `ssh clifford "docker exec kgcos0o4cw4ok0ss0g08wswo psql -U postgres -d stopsign"`
- MinIO access: Use mc client or web console at MinIO endpoint
- All credentials in environment variables (see docker/production/.env on cube's Coolify deployment)