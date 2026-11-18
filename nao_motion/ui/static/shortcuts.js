"use strict";
// Raccourcis clavier de la prise de posture (onglets 1 · Bras et 2 · Tête) :
//   Espace     relâche les membres concernés, partout dans l'interface
//              (bras dans l'onglet Bras, tête dans l'onglet Tête, groupe choisi pendant
//              une calibration ouverte, bras + tête ailleurs) ;
//   1 2 3 4    déclenchent directement les étapes Préparer / Démarrer / Arrêter / Raidir ;
//   →          déclenche l'étape suivante, déduite de l'état de l'enregistrement ;
//   Entrée     valide (« Accepter ») la prise affichée en aperçu, y compris depuis le champ
//              Annotation de la prise (on tape la note puis Entrée) ;
//   Suppr      rejette (« Rejeter ») la prise affichée en aperçu (hors champ de saisie,
//              où Suppr garde son sens habituel) ;
//   0          lance le « Tirage aléatoire » équilibré (émotion, intensité, angle, distance) des
//              onglets Bras, Tête, Mains et Génération. Le tirage ne fait que remplir la
//              consigne et l'événement : aucun mouvement.
//   1 2 3 4    onglet Génération : « Générer et valider » / « Play on the robot » /
//              « Repos puis relâcher » / « Raidir sur place » (sans confirmation, moteurs
//              armés requis pour 2 et 3 ; avertissement permanent #g-alert). Espace y relâche
//              tous les moteurs (bras, tête, mains), même pendant un mouvement.
//   V          inverse gauche/droite du champ de vision (vue robot de dos <-> vue observateur
//              de face), mêmes onglets ; affichage seul, l'événement ne change pas.
// Les touches sont ignorées quand le focus est dans un champ de saisie (texte, nombre,
// liste) et quand Ctrl, Alt ou Cmd sont enfoncés. Chaque étape passe par le bouton de
// l'interface : mêmes garde-fous côté serveur. Relâcher n'est pas
// un mouvement commandé : aucune raideur n'est ajoutée, rien n'est déplacé.
(function () {
  const STEPS = {
    arms: ["a-prepare", "a-start", "a-stop", "a-hold"],
    head: ["h-prepare", "h-start", "h-stop", "h-hold"],
    hands: ["m-prepare", "m-start", "m-stop", "m-hold"],
  };
  const STEP_LABEL = ["Préparer", "Démarrer", "Arrêter", "Raidir"];
  // aperçu de la prise, son bouton Accepter et son champ Annotation, par onglet
  const DRAFT = {
    arms: {box: "a-draft", accept: "a-accept", reject: "a-reject", notes: "a-notes"},
    head: {box: "h-draft", accept: "h-accept", reject: "h-reject", notes: "h-notes"},
    hands: {box: "m-draft", accept: "m-accept", reject: "m-reject", notes: "m-notes"},
  };
  // bouton « Tirage aléatoire » par onglet
  const RANDOM = {arms: "a-random", head: "h-random", hands: "m-random", gen: "g-random"};
  // Generation tab: 1 = generate and validate, 2 = play the last generated gesture,
  // 3 = back to rest then relax, 4 = stiffen everything in place
  const GEN = ["g-go", "g-play", "g-rest-relax", "g-hold"];
  const GEN_LABEL = ["générer et valider", "play on the robot", "repos puis relâcher", "raidir sur place"];
  const ARMS = ["left_arm", "right_arm"], ALL = ["head", "left_arm", "right_arm"];
  const lastHold = {arms: false, head: false, hands: false};
  const bar = document.getElementById("shortcuts");
  // The bar sits under the tabs; in a recording tab it moves into the shortcut slot of the
  // Recording frame (display only, same content).
  const barHome = bar && document.createComment("shortcuts");
  if (bar) bar.parentNode.insertBefore(barHome, bar);
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
    if (tab === "gen") return {label: "tous les moteurs", groups: [...ALL, "hands"]};
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

  // prise en aperçu (enregistrement terminé, pas encore classée) dans l'onglet courant
  function draftReady(kind) {
    const d = kind && DRAFT[kind], box = d && document.getElementById(d.box);
    const btn = d && document.getElementById(d.accept);
    return !!(box && btn && !box.classList.contains("hidden") && !btn.disabled) && btn;
  }

  let accepting = false;
  function accept(kind) {
    const btn = draftReady(kind);
    if (!btn) { flash("Entrée : aucune prise en aperçu à valider."); return; }
    if (accepting) return; // évite un double enregistrement si Entrée est frappée deux fois
    accepting = true;
    flash("Entrée · prise acceptée");
    Promise.resolve(btn.onclick ? btn.onclick() : btn.click())
      .catch(() => flash("Validation refusée (voir le message d'erreur)."))
      .finally(() => { accepting = false; });
  }

  // Suppr : rejette la prise en aperçu, par son bouton « Rejeter » (mêmes garde-fous)
  function reject(kind) {
    const d = draftReady(kind) && document.getElementById(DRAFT[kind].reject);
    if (!d || d.disabled) { flash("Suppr : aucune prise en aperçu à rejeter."); return; }
    if (accepting) return; // une seule décision par prise
    accepting = true;
    flash("Suppr · prise rejetée");
    Promise.resolve(d.onclick ? d.onclick() : d.click())
      .catch(() => flash("Rejet refusé (voir le message d'erreur)."))
      .finally(() => { accepting = false; });
  }

  function randomButton(tab) {
    const btn = RANDOM[tab] && document.getElementById(RANDOM[tab]);
    return btn && !btn.disabled ? btn : null;
  }

  function randomDraw(tab) {
    const btn = randomButton(tab);
    if (!btn) return;
    flash("0 · tirage aléatoire");
    Promise.resolve(btn.onclick ? btn.onclick() : btn.click())
      .catch(() => flash("Tirage refusé (voir le message d'erreur)."));
  }

  function flash(text) { flashText = text; flashUntil = performance.now() + 2500; render(); }

  function render() {
    const tab = activeTab(), kind = tab in STEPS ? tab : null;
    for (const [k, ids] of Object.entries(STEPS)) {
      const n = k === kind ? nextStep(k) : -1;
      ids.forEach((id, i) => { const b = document.getElementById(id); if (b) b.classList.toggle("next", i === n); });
    }
    if (!bar) return;
    const slot = document.querySelector("section.tab.active .sc-slot");
    if (slot && bar.parentNode !== slot) slot.prepend(bar);
    else if (!slot && bar.previousSibling !== barHome) barHome.after(bar);
    const t = releaseTarget();
    let h = `<span><kbd>Espace</kbd> relâcher ${t.label}</span>`;
    if (kind) h += `<span><kbd>1</kbd>–<kbd>4</kbd> étapes</span>` +
      `<span><kbd>→</kbd> étape suivante : <b>${nextStep(kind) + 1} · ${STEP_LABEL[nextStep(kind)]}</b></span>`;
    if (tab === "gen") h += `<span><kbd>1</kbd> générer et valider</span><span><kbd>2</kbd> <b>play on the robot</b></span>` +
      `<span><kbd>3</kbd> repos puis relâcher</span><span><kbd>4</kbd> raidir sur place</span>`;
    if (randomButton(tab)) h += `<span><kbd>0</kbd> tirage aléatoire</span>`;
    if (RANDOM[tab] && window.FOV) h += `<span><kbd>V</kbd> vue ${window.FOV.view() === "observer" ? "observateur" : "robot"}</span>`;
    if (kind && draftReady(kind)) h += `<span><kbd>Entrée</kbd> <b>accepter la prise</b></span>` +
      `<span><kbd>Suppr</kbd> rejeter la prise</span>`;
    h += `<span class="sc-note">inactifs dans un champ de saisie</span>`;
    if (performance.now() < flashUntil) h += `<span class="sc-flash">${flashText}</span>`;
    bar.innerHTML = h;
  }

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
    const tab = activeTab(), kind = tab in STEPS ? tab : null;
    const isEnter = e.key === "Enter" || e.code === "NumpadEnter";
    // Entrée dans le champ Annotation de la prise : on garde la note et on valide.
    if (isEnter && kind && e.target && e.target.id === DRAFT[kind].notes) {
      e.preventDefault(); if (!e.repeat) accept(kind); return;
    }
    if (isEntry(e.target)) return;
    if (e.key === " " || e.code === "Space") {
      e.preventDefault(); // pas de défilement ni de clic sur le bouton qui a le focus
      if (!e.repeat) release();
      return;
    }
    // 0 : e.code couvre la rangée du haut AZERTY (où « 0 » demande Maj) et le pavé numérique
    if (RANDOM[tab] && (e.code === "Digit0" || e.code === "Numpad0" || e.key === "0")) {
      e.preventDefault(); if (!e.repeat) randomDraw(tab); return;
    }
    if (RANDOM[tab] && window.FOV && (e.code === "KeyV" || e.key === "v" || e.key === "V")) {
      e.preventDefault();
      if (!e.repeat) flash(window.FOV.toggleView() === "observer" ? "V · vue observateur (face)" : "V · vue robot (dos)");
      return;
    }
    if (tab === "gen") {
      const g = /^(?:Digit|Numpad)([1-4])$/.exec(e.code) || /^([1-4])$/.exec(e.key);
      if (!g) return;
      e.preventDefault();
      if (e.repeat) return;
      const btn = document.getElementById(GEN[+g[1] - 1]);
      if (!btn || btn.disabled) { flash(`${g[1]} : indisponible` + (g[1] === "2" || g[1] === "3" ? " (voir sous le bouton Play)" : "")); return; }
      flash(`${g[1]} · ${GEN_LABEL[+g[1] - 1]}`);
      btn.click();
      return;
    }
    if (!kind) return;
    // Entrée : jamais le clic natif sur le bouton qui a gardé le focus (ex. « Arrêter »).
    if (isEnter) { e.preventDefault(); if (!e.repeat) accept(kind); return; }
    if (e.key === "Delete" || e.code === "Delete") {
      e.preventDefault(); if (!e.repeat) reject(kind); return;
    }
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
