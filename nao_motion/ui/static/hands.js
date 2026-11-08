"use strict";
// Onglet 3 · Mains : ouverture des doigts de NAO (LHand, RHand), de 0 (fermée) à 1 (ouverte).
// Une main de NAO ne se guide pas à la main : on la pilote d'ici, par les curseurs, les
// boutons ou le clavier (↑ ↓ O F dans l'onglet Mains).
//   - pendant le passage 3 (compte à rebours ou enregistrement), chaque réglage est envoyé
//     en continu et suivi en direct par la boucle d'enregistrement ;
//   - sinon, un réglage = un mouvement court validé par le guardian, envoyé au relâchement
//     du curseur (contrôleur armé requis : « Mouvements autorisés »).
// Les erreurs s'affichent sous les curseurs (pas de boîte de dialogue qui bloquerait le clavier).
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

  // en direct : au plus une requête toutes les 60 ms, la dernière valeur gagne
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
    s.addEventListener("input", () => setHands(linked(j, s.value)));            // en direct pendant le passage 3
    s.addEventListener("change", () => { if (!recording()) setHands(linked(j, s.value), {commit: true}); });
    show(j);
  }
  document.querySelectorAll("[data-hand-set]").forEach((b) => b.addEventListener("click", () => {
    const v = {};
    b.dataset.handSet.split(",").forEach((p) => { const [j, x] = p.split(":"); v[j] = +x; });
    setHands(v, {commit: true});
  }));

  // clavier, dans l'onglet Mains seulement (mêmes règles que shortcuts.js)
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
    e.preventDefault();  // pas de défilement, pas de curseur déplacé deux fois
    setHands(v, {commit: true});
  });

  // mesure et consigne affichées ; curseurs recalés sur la mesure hors saisie
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
