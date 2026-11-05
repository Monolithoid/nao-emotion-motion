"use strict";
// Raccourcis clavier de la prise de posture (onglets 1 · Bras et 2 · Tête) :
//   Espace     relâche les membres concernés, partout dans l'interface
//              (bras dans l'onglet Bras, tête dans l'onglet Tête, groupe choisi pendant
//              une calibration ouverte, bras + tête ailleurs) ;
//   1 2 3 4    déclenchent directement les étapes Préparer / Démarrer / Arrêter / Raidir ;
//   →          déclenche l'étape suivante, déduite de l'état de l'enregistrement.
// Les touches sont ignorées quand le focus est dans un champ de saisie (texte, nombre,
// liste) et quand Ctrl, Alt ou Cmd sont enfoncés. Chaque étape passe par le bouton de
// l'interface : mêmes confirmations, même garde-fous côté serveur. Relâcher n'est pas
// un mouvement commandé : aucune raideur n'est ajoutée, rien n'est déplacé.
(function () {
  const STEPS = {
    arms: ["a-prepare", "a-start", "a-stop", "a-hold"],
    head: ["h-prepare", "h-start", "h-stop", "h-hold"],
  };
  const STEP_LABEL = ["Préparer", "Démarrer", "Arrêter", "Raidir"];
  const ARMS = ["left_arm", "right_arm"], ALL = ["head", "left_arm", "right_arm"];
  const lastHold = {arms: false, head: false};
  const bar = document.getElementById("shortcuts");
  let flashUntil = 0, flashText = "";

  const activeTab = () => { const t = document.querySelector("section.tab.active"); return t ? t.id.slice(4) : ""; };
  const recState = (kind) => {
    const r = (typeof STATUS !== "undefined" && STATUS && STATUS.recording) || {state: "none"};
    return r.kind === kind ? r.state : "none";
  };

  // Champ de saisie : les touches y gardent leur sens habituel. Cases à cocher, boutons
  // radio et curseurs n'en sont pas : sinon le dernier curseur touché avalerait la flèche.
  function isEntry(el) {
    if (!el || el === document.body) return false;
    if (el.isContentEditable) return true;
    const tag = el.tagName;
    if (tag === "TEXTAREA" || tag === "SELECT") return true;
    if (tag !== "INPUT") return false;
    return !["checkbox", "radio", "range", "button", "submit", "reset", "color"].includes((el.type || "").toLowerCase());
  }

  // étape suivante (0..3) pour l'onglet Bras ou Tête
  function nextStep(kind) {
    const s = recState(kind);
    if (s === "prepared") return 1;
    if (s === "countdown" || s === "recording") return 2;
    if (s === "done") return lastHold[kind] ? 0 : 3;
    return 0;
  }

  // ce que relâche Espace dans le contexte courant
  function releaseTarget() {
    const tab = activeTab();
    if (tab === "arms") return {label: "les bras", groups: ARMS};
    if (tab === "head") return {label: "la tête", groups: ["head"]};
    if (tab === "calib" && window.CALIB && window.CALIB.active)
      return {label: `le groupe « ${window.CALIB.groupLabel} » (calibration)`, calib: true};
    return {label: "les bras et la tête", groups: ALL};
  }

  function release() {
    const t = releaseTarget();
    const p = t.calib ? window.CALIB.release() : api("relax", {groups: t.groups, confirm: true});
    flash(`Espace : relâchement (${t.label}), à tenir.`);
    Promise.resolve(p).catch(() => flash("Relâchement refusé (voir le message d'erreur)."));
  }

  function runStep(kind, i) {
    const btn = document.getElementById(STEPS[kind][i]);
    if (!btn || btn.disabled) return;
    flash(`${i + 1} · ${STEP_LABEL[i]}`);
    btn.click();
  }

  function flash(text) { flashText = text; flashUntil = performance.now() + 2500; render(); }

  function render() {
    const tab = activeTab(), kind = tab in STEPS ? tab : null;
    for (const [k, ids] of Object.entries(STEPS)) {
      const n = k === kind ? nextStep(k) : -1;
      ids.forEach((id, i) => { const b = document.getElementById(id); if (b) b.classList.toggle("next", i === n); });
    }
    if (!bar) return;
    const t = releaseTarget();
    let h = `<span><kbd>Espace</kbd> relâcher ${t.label}</span>`;
    if (kind) h += `<span><kbd>1</kbd>–<kbd>4</kbd> étapes</span>` +
      `<span><kbd>→</kbd> étape suivante : <b>${nextStep(kind) + 1} · ${STEP_LABEL[nextStep(kind)]}</b></span>`;
    h += `<span class="sc-note">inactifs dans un champ de saisie</span>`;
    if (performance.now() < flashUntil) h += `<span class="sc-flash">${flashText}</span>`;
    bar.innerHTML = h;
  }

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
    if (isEntry(e.target)) return;
    const tab = activeTab(), kind = tab in STEPS ? tab : null;
    if (e.key === " " || e.code === "Space") {
      e.preventDefault(); // pas de défilement ni de clic sur le bouton qui a le focus
      if (!e.repeat) release();
      return;
    }
    if (!kind) return;
    // chiffres : e.code couvre la rangée du haut AZERTY (où « 1 » demande Maj) et le pavé numérique
    const m = /^(?:Digit|Numpad)([1-4])$/.exec(e.code) || /^([1-4])$/.exec(e.key);
    if (m) { e.preventDefault(); if (!e.repeat) runStep(kind, +m[1] - 1); return; }
    if (e.key === "ArrowRight") { e.preventDefault(); if (!e.repeat) runStep(kind, nextStep(kind)); }
  });

  // « Raidir » (au clavier ou à la souris) clôt le cycle : la flèche repart sur Préparer.
  for (const kind of Object.keys(STEPS)) {
    const [prep, , , hold] = STEPS[kind].map((id) => document.getElementById(id));
    if (hold) hold.addEventListener("click", () => { lastHold[kind] = true; });
    if (prep) prep.addEventListener("click", () => { lastHold[kind] = false; });
  }
  document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => setTimeout(render, 0)));
  setInterval(render, 300);
  render();
})();
