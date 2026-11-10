"use strict";
// Illustration du NAO dans la colonne de droite : montre sur quelle partie du robot porte
// l'action en cours (bras pendant une prise de bras, tête pendant une prise de tête,
// groupe choisi pendant la calibration) et l'état réel de chaque groupe lu dans
// /api/status (roles) : relâché = à guider à la main, raidi = maintenu ou motorisé.
// Purement visuel : rien ici n'appelle l'API.
(function () {
  // Zones dessinées sur nao.png (260 × 300). Le robot fait face à l'observateur :
  // son bras droit est à gauche de l'image (bras tendu), son bras gauche à droite.
  const PARTS = {
    head: {label: "Tête", shape: '<ellipse cx="179" cy="31" rx="41" ry="32"/>', tag: [238, 66]},
    right_arm: {label: "Bras droit",
      shape: '<polygon points="16,12 60,14 92,26 130,28 150,36 162,56 150,76 128,66 96,60 60,62 16,60"/>', tag: [52, 78]},
    left_arm: {label: "Bras gauche",
      shape: '<polygon points="178,64 204,66 232,82 258,104 258,130 226,132 196,118 180,98"/>', tag: [222, 148]},
    // mains (LHand, RHand) : dessinées après les bras, donc au-dessus
    hands: {label: "Mains",
      shape: '<ellipse cx="36" cy="36" rx="24" ry="25"/><ellipse cx="243" cy="117" rx="17" ry="16"/>', tag: [36, 96]},
  };
  const GROUPS = ["head", "right_arm", "left_arm", "hands"];
  const JOINT_GROUP = (j) => (j.startsWith("Head") ? "head" : j.endsWith("Hand") ? "hands" :
    j[0] === "L" ? "left_arm" : j[0] === "R" ? "right_arm" : null);
  const TABS = ["arms", "head", "hands", "gen", "calib", "robot"];
  const box = document.getElementById("nao-fig");
  if (!box) return;

  box.innerHTML =
    `<svg class="nao-svg" viewBox="0 0 260 300" role="img" aria-label="NAO, parties concernées par l'action en cours">` +
    `<image href="nao.png" x="0" y="0" width="260" height="300"/>` +
    GROUPS.map((g) => `<g class="part" data-part="${g}"><title>${PARTS[g].label}</title>${PARTS[g].shape}</g>`).join("") +
    GROUPS.map((g) => `<text class="part-tag" data-tag="${g}" x="${PARTS[g].tag[0]}" y="${PARTS[g].tag[1]}" text-anchor="middle">${PARTS[g].label}</text>`).join("") +
    `</svg>` +
    `<div class="nao-caption" id="nao-caption"></div>` +
    `<div class="nao-legend"><span><i class="sw guided"></i>relâché : à guider</span>` +
    `<span><i class="sw driven"></i>raidi / motorisé</span><span><i class="sw focus"></i>partie concernée</span></div>`;
  box.querySelectorAll(".part").forEach((p) => p.onclick = () => {
    // les mains ne se calibrent pas (plage d'usine 0..1)
    if (activeTab() === "calib" && window.CALIB && p.dataset.part !== "hands") window.CALIB.select(p.dataset.part);
  });

  const activeTab = () => { const t = document.querySelector("section.tab.active"); return t ? t.id.slice(4) : ""; };

  // état réel de chaque groupe : guided si une articulation est relâchée, sinon driven si raidie
  function groupRoles(roles) {
    const out = {head: "idle", left_arm: "idle", right_arm: "idle", hands: "idle"};
    for (const [j, r] of Object.entries(roles || {})) {
      const g = JOINT_GROUP(j);
      if (!g || !(g in out)) continue;
      if (r === "guided") out[g] = "guided";
      else if (r === "driven" && out[g] !== "guided") out[g] = "driven";
    }
    return out;
  }

  // parties concernées, animation et légende selon l'onglet et l'action en cours
  function focusFor(tab, st) {
    const r = (st && st.recording) || {state: "none"};
    const rec = (kind) => (r.kind === kind ? r.state : "none");
    const step = (kind, idle, prepared, recording) => {
      const s = rec(kind);
      if (s === "prepared") return [prepared, false];
      if (s === "countdown") return [`Départ dans ${Math.ceil(r.countdown_left || 0)}… ${recording}`, true];
      if (s === "recording") return [`● Enregistrement : ${recording}`, true];
      if (s === "done") return ["Prise terminée : vérifier l'aperçu, puis accepter ou rejeter.", false];
      return [idle, false];
    };
    if (tab === "arms") {
      const [text, live] = step("arms", "Passage 1 : les deux bras seront relâchés, à tenir et guider ; la tête reste raide.",
        "Bras relâchés : tenez-les. La tête reste raide sur place.", "guidez les bras.");
      return {parts: ["left_arm", "right_arm"], live, text};
    }
    if (tab === "head") {
      const [text, live] = step("head", "Passage tête : la tête sera relâchée et guidée à la main ; les bras restent raides sur place.",
        "Tête relâchée : tenez-la. Les bras restent raides.", "guidez la tête.");
      return {parts: ["head"], live, text};
    }
    if (tab === "hands") {
      const [text, live] = step("hands", "Passage mains : bras et tête restent raides sur place ; vous pilotez l'ouverture des mains d'ici.",
        "Tout est raide sur place, rien n'est relâché. Réglez l'ouverture de départ si besoin.", "ouvrez ou fermez les mains, puis Arrêter.");
      return {parts: ["hands"], live, text};
    }
    if (tab === "calib") {
      const c = window.CALIB;
      if (!c || !c.active) return {parts: [], live: false, text: "Calibration fermée. Une fois ouverte, la partie choisie est mise en évidence (clic sur le robot pour la choisir)."};
      const g = c.sweeping || c.group;
      const lbl = PARTS[g] ? PARTS[g].label.toLowerCase() : g;
      return c.sweeping
        ? {parts: [g], live: true, text: `Exploration en cours : amenez lentement ${g === "head" ? "la tête" : "le " + lbl} jusqu'aux butées utiles.`}
        : {parts: [g], live: false, text: `Groupe choisi : ${lbl}. Cliquer sur une autre partie du robot pour changer de groupe.`};
    }
    if (tab === "gen") return {parts: ["head", "left_arm", "right_arm", "hands"], live: false, text: "Le geste généré anime les bras, la tête et, si le modèle des mains est entraîné, l'ouverture des mains (les jambes ne bougent jamais)."};
    return {parts: [], live: false, text: "État des groupes lu sur le robot : relâché ou raidi."};
  }

  function update() {
    const tab = activeTab();
    document.body.dataset.tab = tab;
    box.classList.toggle("hidden", !TABS.includes(tab));
    if (!TABS.includes(tab)) return;
    const st = typeof STATUS !== "undefined" ? STATUS : null;
    const roles = groupRoles(st && st.roles);
    const f = focusFor(tab, st);
    for (const g of GROUPS) {
      const el = box.querySelector(`[data-part="${g}"]`);
      const on = f.parts.includes(g);
      el.setAttribute("class", `part ${roles[g]}${on ? " focus" : ""}${on && f.live ? " live" : ""}${tab === "calib" && window.CALIB && window.CALIB.active && g !== "hands" ? " pick" : ""}`);
      box.querySelector(`[data-tag="${g}"]`).classList.toggle("on", on || roles[g] !== "idle");
    }
    const c = box.querySelector("#nao-caption");
    if (c.textContent !== f.text) c.textContent = f.text;
  }

  window.NAOFIG = {update};
  update();
})();
