"use strict";
// Tab 3 · Mains: NAO finger opening (LHand, RHand), from 0 (closed) to 1 (open).
// A NAO hand cannot be guided by hand: it is driven from here, with the sliders, the
// buttons or the keyboard (↑ ↓ O F in the Mains tab).
//   - during the hands pass (countdown or recording), each setting is sent
//     continuously and followed live by the recording loop;
//   - otherwise, one setting = one short movement validated by the guardian, sent when
//     the slider is released (armed controller required: "Mouvements autorisés").
// Errors are shown under the sliders (no dialog that would block the keyboard).
(function () {
  const HANDS = ["LHand", "RHand"];
  const STEP = 0.1;
  const slider = (j) => document.getElementById(`m-${j}`);
  const live = document.getElementById("m-live");
  if (!slider("LHand")) return;
  let lastEdit = 0, lastSent = 0, pending = null, timer = null;

  const fmt = (x) => Number(x).toFixed(2).replace(".", ",");
  const activeTab = () => { const t = document.querySelector("section.tab.active"); return t ? t.id.slice(4) : ""; };
  const recording = () => {
    const r = (typeof STATUS !== "undefined" && STATUS && STATUS.recording) || {};
    return r.kind === "hands" && (r.state === "countdown" || r.state === "recording");
  };

  function show(j) { document.getElementById(`m-${j}-val`).textContent = fmt(slider(j).value); }

  async function post(targets) {
    try {
      const r = await fetch("/api/hands", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({targets})});
      const d = await r.json();
      if (!r.ok) live.textContent = "Mains : " + d.error + (d.report ? " " + JSON.stringify(d.report.violations) : "");
      else if (!d.live) {
        live.textContent = d.queued ? "Mains : enchaîné après le mouvement en cours." : "";
        if (d.queued) setTimeout(() => { if (live.textContent.includes("enchaîné")) live.textContent = ""; }, 2000);
      }
    } catch (e) { live.textContent = "Mains : " + e; }
  }

  // live: at most one request every 60 ms, the last value wins
  function send(targets, now) {
    pending = Object.assign(pending || {}, targets);
    const go = () => { timer = null; const t = pending; pending = null; lastSent = performance.now(); if (t) post(t); };
    if (now) { clearTimeout(timer); go(); return; }
    if (!timer) timer = setTimeout(go, Math.max(0, 60 - (performance.now() - lastSent)));
  }

  function setHands(values, opts = {}) {
    lastEdit = performance.now();
    const t = {};
    for (const [j, v] of Object.entries(values)) {
      const x = Math.min(1, Math.max(0, +v));
      slider(j).value = x; show(j); t[j] = x;
    }
    if (opts.commit || recording()) send(t, opts.commit && !recording());
  }

  function linked(j, v) {
    const out = {[j]: v};
    if (document.getElementById("m-link").checked) for (const k of HANDS) out[k] = v;
    return out;
  }

  for (const j of HANDS) {
    const s = slider(j);
    s.addEventListener("input", () => setHands(linked(j, s.value)));            // live during the hands pass
    s.addEventListener("change", () => { if (!recording()) setHands(linked(j, s.value), {commit: true}); });
    show(j);
  }
  document.querySelectorAll("[data-hand-set]").forEach((b) => b.addEventListener("click", () => {
    const v = {};
    b.dataset.handSet.split(",").forEach((p) => { const [j, x] = p.split(":"); v[j] = +x; });
    setHands(v, {commit: true});
  }));

  // keyboard, in the Mains tab only (same rules as shortcuts.js)
  function isEntry(el) {
    if (!el || el === document.body) return false;
    if (el.isContentEditable) return true;
    const tag = el.tagName;
    if (tag === "TEXTAREA" || tag === "SELECT") return true;
    if (tag !== "INPUT") return false;
    return !["checkbox", "radio", "range", "button", "submit", "reset", "color"].includes((el.type || "").toLowerCase());
  }
  document.addEventListener("keydown", (e) => {
    if (activeTab() !== "hands" || e.ctrlKey || e.metaKey || e.altKey || e.isComposing || isEntry(e.target)) return;
    const k = e.key.toLowerCase();
    let v = null;
    if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      const d = e.key === "ArrowUp" ? STEP : -STEP;
      v = {}; for (const j of HANDS) v[j] = Math.round((+slider(j).value + d) * 100) / 100;
    } else if (k === "o" && !e.repeat) v = {LHand: 1, RHand: 1};
    else if (k === "f" && !e.repeat) v = {LHand: 0, RHand: 0};
    if (!v) return;
    e.preventDefault();  // no scrolling, no slider moved twice
    setHands(v, {commit: true});
  });

  // measurement and target shown; sliders synced to the measurement outside input
  async function poll() {
    if (activeTab() !== "hands") return;
    let d;
    try { d = await (await fetch("/api/hands")).json(); } catch (e) { return; }
    for (const j of HANDS) {
      const m = d.measured && d.measured[j], c = d.command && d.command[j];
      document.getElementById(`m-${j}-meas`).textContent =
        c != null ? `consigne ${fmt(c)} → cible ${fmt(d.target[j])}` : m != null ? `mesurée ${fmt(m)}` : "";
      if (m != null && !recording() && performance.now() - lastEdit > 2500) { slider(j).value = m; show(j); }
    }
  }
  setInterval(poll, 400);
})();
