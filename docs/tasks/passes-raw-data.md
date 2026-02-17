# Passes + Raw Tracking Persistence — Task Log

## Status
- Phase 0: Spec ✅
- Phase 1: Raw capture + data model ✅
- Phase 1.5: Review fixes ✅
- Phase 2: Pass detail page + linking ⏳

## Phase 1 Checklist
- [x] Add `vehicle_pass_raw` SQLAlchemy model + migrations via create_all
- [x] Capture raw samples on every tracking update
- [x] Persist raw payload on pass completion (config + model snapshot)
- [x] Add DB helpers (`save_vehicle_pass_raw`, `get_pass_detail`)
- [x] Update task doc

## Phase 1.5 Checklist (review fixes)
- [x] Fix `is_parked` gate killing valid 5+ second stops (in_zone cars exempt)
- [x] Add 1-minute zone timeout (street-parked cars don't generate garbage passes)
- [x] Add 10-minute sliding window on raw samples (bounds memory for long-lived cars)
- [x] Fix detached ORM objects in `get_pass_detail` (expunge before session close)
- [x] Wrap `_build_raw_payload` in try/except for error visibility
- [x] Remove unnecessary `hasattr` guard on `config.get_snapshot`

## Phase 2 Checklist
- [ ] Add `/passes/{id}` page route
- [ ] Add `pass_detail.html` template
- [ ] Link pass items to detail page
- [ ] Update task doc

## Notes
- Raw payload schema v1 uses array samples with `sample_schema` map.
- Coordinate space = processed; include crop/scale + raw dimensions.
- `is_parked` gate was causing zero-pass bug: 4s parked threshold < typical 5s stop.
- Zone timeout (60s) prevents street-parked cars from creating passes with hours of time_in_zone.
- Sample window (600s) prevents memory growth for cars tracked indefinitely by YOLO.
