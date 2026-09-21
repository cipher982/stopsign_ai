/**
 * Lookout — arm a watch from an exact frame and show what it reported.
 *
 * Two rules the UI keeps, both from review:
 *  - Boxes are drawn on a frozen clean frame, not on the live player: the HLS
 *    stream runs ~19s behind capture, so arming against it would select a moment
 *    that is already gone.
 *  - Coordinates sent to the server are in the snapshot's own pixel space, taken
 *    from the image itself, with the letterboxing accounted for. Nothing is
 *    hardcoded to a resolution.
 */
(function () {
    'use strict';

    var state = {
        box: null,
        frame: null,
        frameAge: 0,
        drawing: false,
        start: null,
        seenEvents: null,
        lastAlertId: null
    };

    function $(id) { return document.getElementById(id); }

    function setStatus(message, bad) {
        var el = $('armStatus');
        if (!el) return;
        el.textContent = message;
        el.className = 'lookout-status mono' + (bad ? ' bad' : '');
    }

    // ---------------------------------------------------------------- frame
    function refreshFrame() {
        var img = $('snapshot');
        var empty = $('stageEmpty');
        if (!img) return;
        setStatus('loading frame…');
        fetch('/api/lookout/frame', { credentials: 'same-origin' })
            .then(function (response) {
                if (!response.ok) throw new Error('frame ' + response.status);
                var width = parseInt(response.headers.get('X-Frame-Width') || '0', 10);
                var height = parseInt(response.headers.get('X-Frame-Height') || '0', 10);
                var age = parseFloat(response.headers.get('X-Frame-Age') || '0');
                var ts = parseFloat(response.headers.get('X-Frame-Capture-Ts') || '0');
                return response.blob().then(function (blob) {
                    return { blob: blob, width: width, height: height, age: age, ts: ts };
                });
            })
            .then(function (result) {
                var url = URL.createObjectURL(result.blob);
                img.onload = function () {
                    if (state.frame && state.frame.url) URL.revokeObjectURL(state.frame.url);
                    state.frame = { url: url, width: result.width || img.naturalWidth, height: result.height || img.naturalHeight, ts: result.ts };
                    sizeCanvas();
                    if (empty) empty.classList.add('hidden');
                    setStatus(state.box ? 'box set — name it and start watching' : 'draw a box on the frame');
                };
                img.src = url;
                state.frameAge = result.age;
                $('frameAge').textContent = 'frame: ' + result.age.toFixed(1) + 's old';
                if (result.age > 5) {
                    setStatus('this frame is ' + result.age.toFixed(0) + 's old — take a new one before arming', true);
                }
            })
            .catch(function (error) {
                setStatus('cannot fetch a clean frame: ' + error.message, true);
                if (empty) { empty.textContent = 'no clean frame available'; empty.classList.remove('hidden'); }
            });
    }

    function sizeCanvas() {
        var img = $('snapshot');
        var canvas = $('drawLayer');
        if (!img || !canvas || !img.naturalWidth) return;
        var rect = img.getBoundingClientRect();
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        drawBox();
    }

    // ---------------------------------------------------------------- draw
    function drawBox() {
        var canvas = $('drawLayer');
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!state.box) return;
        var b = state.box;
        ctx.lineWidth = Math.max(2, canvas.width / 500);
        ctx.strokeStyle = '#e14fae';
        ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
        ctx.fillStyle = 'rgba(225,79,174,0.14)';
        ctx.fillRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    }

    function pointerToFrame(event) {
        var canvas = $('drawLayer');
        var rect = canvas.getBoundingClientRect();
        var scaleX = canvas.width / rect.width;
        var scaleY = canvas.height / rect.height;
        return {
            x: Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * scaleX)),
            y: Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * scaleY))
        };
    }

    function wireDrawing() {
        var canvas = $('drawLayer');
        if (!canvas) return;
        canvas.addEventListener('mousedown', function (event) {
            event.preventDefault();
            state.drawing = true;
            state.start = pointerToFrame(event);
            state.box = { x1: state.start.x, y1: state.start.y, x2: state.start.x, y2: state.start.y };
            drawBox();
        });
        canvas.addEventListener('mousemove', function (event) {
            if (!state.drawing) return;
            var point = pointerToFrame(event);
            state.box = {
                x1: Math.min(state.start.x, point.x),
                y1: Math.min(state.start.y, point.y),
                x2: Math.max(state.start.x, point.x),
                y2: Math.max(state.start.y, point.y)
            };
            drawBox();
        });
        window.addEventListener('mouseup', function () {
            if (!state.drawing) return;
            state.drawing = false;
            finishBox();
        });
        canvas.addEventListener('touchstart', handleTouch, { passive: false });
        canvas.addEventListener('touchmove', handleTouch, { passive: false });
        canvas.addEventListener('touchend', function () { state.drawing = false; finishBox(); });
    }

    function handleTouch(event) {
        if (!event.touches.length) return;
        event.preventDefault();
        var touch = event.touches[0];
        var synthetic = { clientX: touch.clientX, clientY: touch.clientY };
        if (!state.drawing) {
            state.drawing = true;
            state.start = pointerToFrame(synthetic);
            state.box = { x1: state.start.x, y1: state.start.y, x2: state.start.x, y2: state.start.y };
        } else {
            var point = pointerToFrame(synthetic);
            state.box = {
                x1: Math.min(state.start.x, point.x),
                y1: Math.min(state.start.y, point.y),
                x2: Math.max(state.start.x, point.x),
                y2: Math.max(state.start.y, point.y)
            };
        }
        drawBox();
    }

    function finishBox() {
        if (!state.box) return;
        var width = state.box.x2 - state.box.x1;
        var height = state.box.y2 - state.box.y1;
        if (width < 24 || height < 24) {
            state.box = null;
            drawBox();
            setStatus('box too small — drag a wider area', true);
            return;
        }
        var button = $('armWatch');
        if (button) button.disabled = false;
        setStatus('box ' + Math.round(width) + '×' + Math.round(height) + ' px — name it and start watching');
    }

    // ---------------------------------------------------------------- arm
    function arm() {
        if (!state.box) { setStatus('draw a box first', true); return; }
        var button = $('armWatch');
        button.disabled = true;
        var payload = {
            box: [state.box.x1, state.box.y1, state.box.x2, state.box.y2],
            condition: $('watchCondition').value,
            label: $('watchLabel').value || 'watch',
            subject: $('watchSubject').value,
            armed: $('watchArmed').value
        };
        setStatus('arming…');
        fetch('/api/lookout/watches', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(function (response) {
                return response.json().then(function (body) {
                    if (!response.ok) throw new Error(body.detail || ('HTTP ' + response.status));
                    return body;
                });
            })
            .then(function (body) {
                setStatus('watching "' + body.watch.label + '" — it will report through the feed below');
                state.box = null;
                drawBox();
                refreshFrame();
                poll();
            })
            .catch(function (error) {
                setStatus('could not arm: ' + error.message, true);
                button.disabled = false;
            });
    }

    function stopWatch(watchId) {
        fetch('/api/lookout/watches/' + encodeURIComponent(watchId) + '/stop', {
            method: 'POST',
            credentials: 'same-origin'
        }).then(poll);
    }

    // ---------------------------------------------------------------- render
    var STATE_CLASS = {
        observing: 'state-observing', occupied: 'state-occupied', gone: 'state-gone',
        unknown: 'state-unknown', unavailable: 'state-unavailable', pending: 'state-pending', stopped: 'state-pending'
    };

    function renderWatches(watches) {
        var list = $('watchList');
        var empty = $('watchEmpty');
        if (!list) return;
        list.textContent = '';
        if (!watches.length) {
            if (empty) empty.classList.remove('hidden');
            return;
        }
        if (empty) empty.classList.add('hidden');
        watches.forEach(function (watch) {
            var status = watch.status || {};
            var card = document.createElement('div');
            card.className = 'lookout-card ' + (STATE_CLASS[status.state] || 'state-pending');

            var head = document.createElement('div');
            head.className = 'lookout-card-head';
            var label = document.createElement('span');
            label.className = 'lookout-card-label';
            label.textContent = watch.label;
            var chip = document.createElement('span');
            chip.className = 'lookout-card-state';
            chip.textContent = status.state || 'arming';
            head.appendChild(label);
            head.appendChild(chip);
            card.appendChild(head);

            var question = document.createElement('div');
            question.className = 'lookout-card-question mono';
            question.textContent = (LOOKOUT_CONDITIONS[watch.condition] || [null, watch.condition])[1];
            card.appendChild(question);

            var metrics = document.createElement('div');
            metrics.className = 'lookout-card-metrics';
            var meter = document.createElement('div');
            meter.className = 'lookout-meter';
            var fill = document.createElement('span');
            fill.style.width = Math.round((status.present_prob || 0) * 100) + '%';
            meter.appendChild(fill);
            var pct = document.createElement('span');
            pct.className = 'lookout-card-pct mono';
            pct.textContent = Math.round((status.present_prob || 0) * 100) + '%';
            metrics.appendChild(meter);
            metrics.appendChild(pct);
            card.appendChild(metrics);

            var reason = document.createElement('div');
            reason.className = 'lookout-card-reason';
            reason.textContent = status.reason || 'waiting for the first reading';
            card.appendChild(reason);

            if (status.history && status.history.length > 1) {
                card.appendChild(sparkline(status.history));
            }

            var foot = document.createElement('div');
            foot.className = 'lookout-card-foot';
            var meta = document.createElement('span');
            meta.className = 'mono lookout-card-question';
            meta.textContent = 'evidence ' + Math.round((status.evidence_fraction || 0) * 100) + '%'
                + (status.alerting_events ? ' · ' + status.alerting_events + ' report(s)' : '');
            var stop = document.createElement('button');
            stop.className = 'lookout-btn ghost';
            stop.type = 'button';
            stop.textContent = 'Stop';
            stop.addEventListener('click', function () { stopWatch(watch.id); });
            foot.appendChild(meta);
            foot.appendChild(stop);
            card.appendChild(foot);

            list.appendChild(card);
        });
    }

    function sparkline(history) {
        var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        var width = 240, height = 30;
        svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
        svg.setAttribute('class', 'lookout-spark');
        svg.setAttribute('preserveAspectRatio', 'none');
        var points = history.map(function (value, index) {
            var x = (index / Math.max(1, history.length - 1)) * width;
            var y = height - Math.max(0, Math.min(1, value)) * height;
            return x.toFixed(1) + ',' + y.toFixed(1);
        }).join(' ');
        var line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        line.setAttribute('points', points);
        line.setAttribute('fill', 'none');
        line.setAttribute('stroke', '#e14fae');
        line.setAttribute('stroke-width', '1.5');
        svg.appendChild(line);
        return svg;
    }

    function renderEvents(events) {
        var feed = $('eventFeed');
        var empty = $('eventEmpty');
        if (!feed) return;
        var interesting = events.filter(function (event) { return !(event.detail && event.detail.note); });
        feed.textContent = '';
        if (!interesting.length) {
            if (empty) empty.classList.remove('hidden');
            return;
        }
        if (empty) empty.classList.add('hidden');
        var newest = null;
        interesting.slice(0, 12).forEach(function (event) {
            if (!newest) newest = event;
            var row = document.createElement('div');
            row.className = 'lookout-event';

            var figure = document.createElement('div');
            var image = document.createElement('img');
            image.loading = 'lazy';
            image.alt = 'Evidence for ' + event.label;
            image.src = (event.evidence && event.evidence.strip) || '';
            figure.appendChild(image);

            var body = document.createElement('div');
            var head = document.createElement('div');
            head.className = 'lookout-event-head';
            var label = document.createElement('span');
            label.className = 'lookout-event-label';
            label.textContent = event.label;
            var stamp = document.createElement('span');
            stamp.className = 'lookout-stamp ' + (event.alerting ? 'alert' : 'note');
            stamp.textContent = event.alerting ? 'reported' : 'note';
            var wording = document.createElement('span');
            wording.className = 'mono';
            wording.textContent = event.wording || event.to_state;
            head.appendChild(label);
            head.appendChild(stamp);
            head.appendChild(wording);
            body.appendChild(head);

            var meta = document.createElement('div');
            meta.className = 'lookout-event-meta';
            var when = event.capture_ts ? new Date(event.capture_ts * 1000).toLocaleString() : '—';
            var lines = [
                when + ' · ' + event.from_state + ' → ' + event.to_state,
                'confidence ' + Math.round((event.present_prob || 0) * 100) + '% · evidence '
                    + Math.round((event.evidence_fraction || 0) * 100) + '% of the window'
                    + (event.detail && event.detail.occupancy_check === 'appearance'
                        ? ' · emptiness not checked for objects' : ''),
                event.detail && event.detail.reason ? event.detail.reason : ''
            ].filter(Boolean).join('\n');
            meta.textContent = lines;
            meta.style.whiteSpace = 'pre-line';
            body.appendChild(meta);

            if (event.evidence && event.evidence.full) {
                var link = document.createElement('a');
                link.className = 'lookout-evidence-link';
                link.href = event.evidence.full;
                link.target = '_blank';
                link.rel = 'noopener';
                link.textContent = 'open the frame that decided it →';
                body.appendChild(link);
            }
            row.appendChild(figure);
            row.appendChild(body);
            feed.appendChild(row);
        });

        if (newest && newest.alerting && newest.event_id !== state.lastAlertId) {
            if (state.lastAlertId !== null) announce(newest);
            state.lastAlertId = newest.event_id;
        } else if (newest && !newest.alerting) {
            state.lastAlertId = null;
        }
    }

    function announce(event) {
        var banner = document.createElement('div');
        banner.className = 'lookout-event';
        banner.style.borderColor = 'var(--bad)';
        banner.textContent = 'Lookout: ' + event.label + ' — ' + (event.wording || event.to_state);
        var feed = $('eventFeed');
        if (feed && feed.firstChild) feed.insertBefore(banner, feed.firstChild);
        beep();
    }

    function beep() {
        try {
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.frequency.value = 660;
            gain.gain.value = 0.06;
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            setTimeout(function () { osc.stop(); ctx.close(); }, 180);
        } catch (error) { /* sound is a nicety, never a requirement */ }
    }

    // ---------------------------------------------------------------- polling
    function poll() {
        fetch('/api/lookout/state', { credentials: 'same-origin' })
            .then(function (response) {
                if (!response.ok) throw new Error('state ' + response.status);
                return response.json();
            })
            .then(function (body) {
                renderWatches(body.watches || []);
                renderEvents(body.events || []);
            })
            .catch(function () { /* keep the last good view */ });
    }

    document.addEventListener('DOMContentLoaded', function () {
        wireDrawing();
        refreshFrame();
        poll();
        window.setInterval(poll, 2500);
        var refresh = $('refreshFrame');
        if (refresh) refresh.addEventListener('click', refreshFrame);
        var clear = $('clearBox');
        if (clear) clear.addEventListener('click', function () {
            state.box = null;
            drawBox();
            $('armWatch').disabled = true;
            setStatus('draw a box on the frame');
        });
        var armButton = $('armWatch');
        if (armButton) armButton.addEventListener('click', arm);
        window.addEventListener('resize', sizeCanvas);
    });
})();
