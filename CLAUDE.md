# Stop Sign AI - Zone Configuration

## Zone Editing Philosophy

**Consistent UX**:
- Stop zone: 4 clicks = rectangle (area detection)
- Pre-stop & capture: 2 clicks = line (crossing detection)

**Unified Naming**:
- Stop detection area is always stored as `stop_zone`
- Exactly four corner points define the polygon
- No tolerance or alternate representations

## Zone Types

1. **Stop Zone**: Rectangle where vehicles must stop
   - Format: `[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]` (raw coordinates, clockwise order)
   - Vehicles are tracked while inside this polygon

2. **Pre-Stop Line**: 2-point line for approach detection
   - Format: `[[x1, y1], [x2, y2]]` in raw coordinates
   - Click 2 points to define the line (handles road angle)
   - Vehicles cross this line before entering stop zone

3. **Capture Line**: 2-point line for photo trigger
   - Format: `[[x1, y1], [x2, y2]]` in raw coordinates
   - Click 2 points to define the line
   - Triggers photo when vehicle crosses

## Real-time Zone Editing

Navigate to `/debug` to edit zones:
1. Select zone type (stop-line, pre-stop, capture)
2. Click "Adjust" to enter edit mode
3. Click the required number of points (4 for stop zone, 2 for line zones)
4. Click "Submit" to save

The browser sends display coordinates, the API converts them to the appropriate coordinate space, and the config is updated. The analyzer automatically reloads when it detects config changes.

## Coordinate Systems

- **Display**: Browser video element pixels
- **Raw**: Original video frame pixels (e.g., 1920x1080)
- **Processing**: After crop/scale (e.g., 1440x810)

Stop zones use raw coordinates. Pre-stop and capture lines use processing coordinates.

## Important: Avoid Overengineering

- Don't reintroduce tolerance math or alternative formats
- Don't add abstraction layers for coordinate math
- Don't create migration scripts for config changes
- Don't maintain parallel data representations

The current implementation is ~400 lines of straightforward code. Keep it that way.
