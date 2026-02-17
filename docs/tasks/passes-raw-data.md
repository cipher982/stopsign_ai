# Passes + Raw Tracking Persistence — Task Log

## Status
- Phase 0: Spec ✅
- Phase 1: Raw capture + data model ⏳
- Phase 2: Pass detail page + linking ⏳

## Phase 1 Checklist
- [ ] Add `vehicle_pass_raw` SQLAlchemy model + migrations via create_all
- [ ] Capture raw samples on every tracking update
- [ ] Persist raw payload on pass completion (config + model snapshot)
- [ ] Add DB helpers (`save_vehicle_pass_raw`, `get_pass_detail`)
- [ ] Update task doc

## Phase 2 Checklist
- [ ] Add `/passes/{id}` page route
- [ ] Add `pass_detail.html` template
- [ ] Link pass items to detail page
- [ ] Update task doc

## Notes
- Raw payload schema v1 uses array samples with `sample_schema` map.
- Coordinate space = processed; include crop/scale + raw dimensions.

