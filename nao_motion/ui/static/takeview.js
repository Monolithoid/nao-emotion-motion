"use strict";
// Takes tab: viewer of the selected take, right of the list. One small plot per joint with
// the recorded curve, the B-spline fit the model is trained on (N control points) and,
// on demand, what the active model generates for the same emotion, intensity and event.
// Read only: nothing here moves the robot.

let takeView = null;            // last /api/takes/<id>/view answer
const TV_FIT_WARN = 0.03;       // rad: fit error above which the representation visibly smooths the take

function tvUnit(group) { return group === "hands" ? "" : " rad"; }
const tvFmt = (x, nd = 3) => (x == null ? "—" : Number(x).toFixed(nd).replace(".", ","));
const tvDeg = (r) => `${Math.round(r * 180 / Math.PI)}°`;

function tvEvent(m) {
  const e = m.event;
  if (!e.present) return "no event";
  return `event az ${tvDeg(e.azimuth_rad)} el ${tvDeg(e.elevation_rad)}` + (e.distance_m != null ? ` ${(+e.distance_m).toFixed(2).replace(".", ",")} m` : "");
}

// one joint: recorded curve, fit, control points, model output, segment, cursor
function tvPlot(j, k, v, opts) {
  const W = 440, H = opts.tall ? 190 : 112, PL = 40, PR = 6, PT = 16, PB = 14;
  const raw = v.q.map((r) => r[k]), fit = v.fit.q.map((r) => r[k]);
  const ctrl = v.fit.ctrl.map((r) => r[k]);
  const mod = opts.model && v.model && !v.model.error ? v.model : null;
  const modQ = mod ? mod.q.map((r) => r[k]) : [];
  let lo = Math.min(...raw), hi = Math.max(...raw);
  if (opts.fit) { lo = Math.min(lo, ...fit); hi = Math.max(hi, ...fit); }
  if (opts.ctrl) { lo = Math.min(lo, ...ctrl); hi = Math.max(hi, ...ctrl); }
  if (mod) { lo = Math.min(lo, ...modQ); hi = Math.max(hi, ...modQ); }
  if (hi - lo < 0.02) { const c = (hi + lo) / 2; lo = c - 0.01; hi = c + 0.01; }
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const t0 = v.t[0], t1 = Math.max(v.t[v.t.length - 1], mod ? mod.t[mod.t.length - 1] : 0);
  const X = (t) => PL + (t - t0) / ((t1 - t0) || 1) * (W - PL - PR);
  const Y = (y) => PT + (hi - y) / (hi - lo) * (H - PT - PB);
  const line = (ts, ys, cls) => `<polyline class="${cls}" points="${ys.map((y, i) => `${X(ts[i]).toFixed(1)},${Y(y).toFixed(1)}`).join(" ")}"/>`;
  const color = COLORS[k % COLORS.length];
  const fr = v.fit.rmse_per_joint[j], mr = mod ? mod.rmse_per_joint[j] : null;
  let s = `<svg viewBox="0 0 ${W} ${H}" class="tv-svg" role="img" aria-label="${j}">`;
  const [s0, s1] = v.segment;
  s += `<rect class="tv-seg" x="${X(s0).toFixed(1)}" y="${PT}" width="${(X(s1) - X(s0)).toFixed(1)}" height="${H - PT - PB}"/>`;
  s += `<text class="tv-ax" x="${PL - 4}" y="${PT + 8}" text-anchor="end">${hi.toFixed(2)}</text><text class="tv-ax" x="${PL - 4}" y="${H - PB}" text-anchor="end">${lo.toFixed(2)}</text>`;
  s += `<text class="tv-ax" x="${W - PR}" y="${H - 2}" text-anchor="end">${(t1 - t0).toFixed(1)} s</text>`;
  s += `<text class="tv-name" x="${PL}" y="11" fill="${color}">${j}</text>`;
  s += `<text class="tv-ax" x="${W - PR}" y="11" text-anchor="end"><tspan class="${opts.fit && fr > TV_FIT_WARN ? "tv-warn" : ""}">fit ${tvFmt(fr)}</tspan>${mr != null ? ` · model ${tvFmt(mr)}` : ""}</text>`;
  s += line(v.t, raw, "tv-raw").replace("<polyline", `<polyline stroke="${color}"`);
  if (opts.fit) s += line(v.fit.t, fit, "tv-fit");
  if (opts.ctrl) {
    s += line(v.fit.ctrl_t, ctrl, "tv-poly");
    s += ctrl.map((y, i) => `<circle class="tv-pt" cx="${X(v.fit.ctrl_t[i]).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="2.6"><title>control point ${i + 1}: ${y.toFixed(3)}</title></circle>`).join("");
  }
  if (mod) s += line(mod.t, modQ, "tv-model");
  s += `<line class="tv-cursor" x1="-10" x2="-10" y1="${PT}" y2="${H - PB}"/></svg>`;
  return {html: s, X};
}

// "velocity ×3 (RShoulderPitch, RElbowYaw), acceleration ×1 (RShoulderPitch)"
function tvViolations(vs) {
  const by = {};
  (vs || []).forEach((v) => { const b = by[v.code] = by[v.code] || {n: 0, j: new Set()}; b.n++; if (v.joint) b.j.add(v.joint); });
  return Object.entries(by).map(([c, b]) => `${c} ×${b.n}${b.j.size ? ` (${[...b.j].join(", ")})` : ""}`).join(", ") || "no detail";
}

function renderTakeView() {
  const v = takeView;
  if (!v || !currentTake) return;
  const opts = {fit: $("t-show-fit").checked, ctrl: $("t-show-ctrl").checked && $("t-show-fit").checked,
                model: $("t-show-model").checked};
  const J = v.joints.length, rows = Math.ceil(J / 2);
  const box = $("t-plots");
  box.style.gridTemplateRows = `repeat(${rows}, auto)`;
  const plots = v.joints.map((j, k) => tvPlot(j, k, v, {...opts, tall: J <= 2}));
  box.innerHTML = plots.map((p) => `<div class="tv-cell">${p.html}</div>`).join("");
  box._X = plots.map((p) => p.X);
  // legend + summary
  const f = v.fit, u = tvUnit(v.group);
  const cmp = f.compare.map((c) => `<button class="tv-n${c.n_ctrl === f.n_ctrl ? " sel" : ""}" data-n="${c.n_ctrl}" title="max error ${tvFmt(c.max_err_rad)}${u}">${c.n_ctrl}${c.n_ctrl === f.default_n_ctrl ? "*" : ""} → ${tvFmt(c.rmse_rad)}</button>`).join("");
  let h = `<div class="tv-legend"><span class="lg-raw">━ recorded</span>` +
    (opts.fit ? ` <span class="lg-fit">┅ fit (${f.n_ctrl} points)</span>` : "") +
    (opts.ctrl ? ` <span class="lg-pt">● control points</span>` : "") +
    (opts.model ? ` <span class="lg-model">━ active model</span>` : "") +
    ` <span class="lg-seg">▮ useful segment</span></div>`;
  h += `<div>Fit with <b>${f.n_ctrl}</b> points${f.n_ctrl === f.default_n_ctrl ? ` (default, from ${f.default_source})` : ` (default ${f.default_n_ctrl})`}: ` +
    `RMSE <b class="${f.rmse_rad > TV_FIT_WARN ? "tv-warn" : ""}">${tvFmt(f.rmse_rad)}${u}</b>, max ${tvFmt(f.max_err_rad)}${u}, ` +
    `on ${f.n_samples} samples (${tvFmt(f.sample_rate_hz, 0)} Hz over ${tvFmt(v.duration_s, 2)} s).</div>`;
  h += `<div class="tv-cmp"><span class="legend">RMSE by number of points:</span> ${cmp}</div>`;
  if (opts.model) {
    const m = v.model;
    if (!m) h += `<div class="muted">Loading the active model…</div>`;
    else if (m.error) h += `<div class="muted">Active model: ${m.error}</div>`;
    else {
      h += `<div>Active model <code>${m.run}</code>, same emotion, intensity and event: RMSE <b>${tvFmt(m.rmse_rad)}${u}</b> against this take ` +
        `(its duration ${tvFmt(m.duration_s, 2)} s, take ${tvFmt(v.duration_s, 2)} s; compared on the same phase).</div>`;
      h += `<div class="tv-why">${f.rmse_rad < 0.4 * m.rmse_rad
        ? `The ${f.n_ctrl} points reproduce this take well (fit ${tvFmt(f.rmse_rad)}): most of the gap comes from the <b>model</b>, which averages the takes it learnt, not from the number of points.`
        : `The fit is already far from this take: the <b>number of points</b> limits what the model can learn here; compare with more points above.`}</div>`;
    }
  } else if (f.rmse_rad > TV_FIT_WARN) {
    h += `<div class="tv-why">With ${f.n_ctrl} points the fit visibly smooths this take: try more points above to see how many it needs.</div>`;
  }
  $("t-fitsum").innerHTML = h;
  $("t-fitsum").querySelectorAll("button[data-n]").forEach((b) => b.onclick = () => { $("t-n").value = b.dataset.n; loadTakeView(); });
}

async function loadTakeView() {
  if (!currentTake) return;
  const id = currentTake.meta.take_id, n = $("t-n").value;
  const withModel = $("t-show-model").checked;
  const v = await api(`takes/${id}/view?n=${n || ""}${withModel ? "&model=1" : ""}`);
  if (!currentTake || currentTake.meta.take_id !== id) return;  // another take was opened meanwhile
  takeView = v;
  const sel = $("t-n");
  if (!sel.options.length || sel.dataset.def !== String(v.fit.default_n_ctrl)) {
    const ns = [4, 5, 6, 8, 10, 12, 16, 20, 24, 32];
    if (!ns.includes(v.fit.default_n_ctrl)) ns.push(v.fit.default_n_ctrl);
    sel.innerHTML = ns.sort((a, b) => a - b).map((k) => `<option value="${k}">${k}${k === v.fit.default_n_ctrl ? " *" : ""}</option>`).join("");
    sel.dataset.def = String(v.fit.default_n_ctrl);
  }
  sel.value = String(v.fit.n_ctrl);
  renderTakeView();
}

function clearTakeView() {
  currentTake = null; takeView = null;
  $("t-detail").classList.add("hidden"); $("t-pick").classList.remove("hidden");
  document.querySelectorAll("#t-table tr.open").forEach((tr) => tr.classList.remove("open"));
}

async function openTake(id) {
  const keepN = currentTake && takeView && currentTake.meta.kind === (TAKES.find((x) => x.take_id === id) || {}).kind
    && takeView.fit.n_ctrl !== takeView.fit.default_n_ctrl ? $("t-n").value : "";
  currentTake = await api("takes/" + id);
  const m = currentTake.meta;
  takeView = null;
  $("t-pick").classList.add("hidden");
  $("t-detail").classList.remove("hidden");
  document.querySelectorAll("#t-table tr[data-id]").forEach((tr) => tr.classList.toggle("open", tr.dataset.id === id));
  const kindLabel = {arms: "arms", head: "head", hands: "hands"}[m.kind] || m.kind;
  $("t-title").textContent = `${label(m.emotion)} ${fmtInt(m.intensity)} · ${kindLabel}`;
  const row = TAKES.find((x) => x.take_id === id) || {};
  const chips = [
    `<span class="tag ${m.status}">${m.status}</span>`,
    row.in_export ? `<span class="tag accepted" title="Published by export-dataset">exported</span>` : `<span class="tag muted" title="Not published by export-dataset">not exported</span>`,
    m.synthetic ? `<span class="tag">simulated</span>` : "",
    isIncomplete(m) ? `<span class="tag">incomplete</span>` : "",
    `<span class="tag">${tvEvent(m)}</span>`,
    `<span class="tag">${(m.segment[1] - m.segment[0]).toFixed(2).replace(".", ",")} s useful / ${m.duration_s.toFixed(1).replace(".", ",")} s</span>`,
    m.intention ? `<span class="tag">intention ${m.intention}</span>` : "",
    `<span class="tag muted mono-id">${m.take_id}</span>`,
  ];
  $("t-chips").innerHTML = chips.join(" ") + `<span id="t-role"></span>`;
  $("t-notes").value = m.annotations.notes || "";
  $("t-plots").innerHTML = "";
  $("t-fitsum").textContent = "";
  if (keepN) $("t-n").value = keepN; else $("t-n").value = "";
  const ev = m.event.present ? `az ${(m.event.azimuth_rad * 180 / Math.PI).toFixed(1)}°, el ${(m.event.elevation_rad * 180 / Math.PI).toFixed(1)}°` +
    (m.event.distance_m != null ? `, ${(+m.event.distance_m).toFixed(2)} m` : ", distance unknown") + " (torso frame)" : "no event";
  $("t-info").textContent = `session ${m.session_id} · ${m.n_samples} samples · ${m.sample_rate_hz} Hz · segment ${m.segment.map((x) => x.toFixed(2)).join("–")} s · calibration ${m.calibration_version}\n` +
    `guided: ${m.guided_joints.join(", ")} · driven: ${m.driven_joints.join(", ") || "—"} · ${ev}` +
    (m.arm_take_id || m.arm_source ? `\nolder take recorded over replayed arms${m.arm_take_id ? ": " + m.arm_take_id : " (generated gesture)"}; only the ${m.kind === "head" ? "head" : "hand"} joints are used for training` : "") +
    (currentTake.head_takes.length ? `\n${currentTake.head_takes.length} head take(s) over this movement` : "") +
    ((currentTake.hand_takes || []).length ? `\n${currentTake.hand_takes.length} hand take(s) over this movement` : "") +
    (currentTake.replay ? `\nreplay (guardian): ${currentTake.replay.guardian.ok ? "accepted" : "refused, " + tvViolations(currentTake.replay.guardian.violations)}` : "");
  await loadTakeView();
  if (takeView && takeView.role) $("t-role").innerHTML = ` <span class="tag" title="Role of this take in the active model ${takeView.active_run}">model: ${takeView.role}</span>`;
}

function playTakeView() {
  const box = $("t-plots");
  if (!takeView || !box._X) return;
  const cursors = [...box.querySelectorAll(".tv-cursor")];
  const t0 = takeView.t[0];
  const mod = $("t-show-model").checked && takeView.model && !takeView.model.error ? takeView.model : null;
  const t1 = Math.max(takeView.t[takeView.t.length - 1], mod ? mod.t[mod.t.length - 1] : 0);
  const start = performance.now();
  const step = (now) => {
    const t = t0 + (now - start) / 1000;
    cursors.forEach((c, k) => { const x = box._X[k](Math.min(t, t1)); c.setAttribute("x1", x); c.setAttribute("x2", x); });
    if (t < t1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function initTakeView() {
  $("t-n").onchange = loadTakeView;
  $("t-show-fit").onchange = renderTakeView;
  $("t-show-ctrl").onchange = renderTakeView;
  $("t-show-model").onchange = () => (takeView && !takeView.model ? loadTakeView() : renderTakeView());
  $("t-play").onclick = playTakeView;
  const upd = (body) => api("takes/" + currentTake.meta.take_id, body).then(() => { loadTakes(); openTake(currentTake.meta.take_id); });
  $("t-accept").onclick = () => currentTake && upd({status: "accepted"});
  $("t-reject").onclick = () => currentTake && upd({status: "rejected"});
  $("t-save-notes").onclick = () => currentTake && upd({notes: $("t-notes").value});
}
