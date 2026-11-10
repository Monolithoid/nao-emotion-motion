"use strict";
// Onglet Calibration : conduite pas à pas de la calibration du guardian.
// La page ne garde aucun état propre : elle redessine tout depuis /api/calibration.
// Aucune route appelée ici ne commande de mouvement.
(function () {
  const GROUP_LABEL = {head: "Tête", left_arm: "Bras gauche", right_arm: "Bras droit"};
  const POSE_LABEL = {rest: "repos", start: "départ"};
  let CAL = {active: false}, group = "head", busy = false;
  const glabel = (g) => GROUP_LABEL[g] || g;
  const f3 = (v) => (typeof v === "number" ? v.toFixed(3) : "—");
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

  async function call(path, body) {
    if (busy) return;
    busy = true;
    try { CAL = await api("calibration/" + path, body || {}); } catch (e) { /* api() affiche l'erreur */ }
    finally { busy = false; render(); }
  }
  const confirmRelease = () => !real() ||
    window.confirm(`Le groupe « ${glabel(group)} » va être RELÂCHÉ (raideur nulle). Le tenez-vous ?`);

  function render() {
    const on = !!CAL.active;
    $("c-body").classList.toggle("hidden", !on);
    $("c-open").disabled = on; $("c-close").disabled = !on;
    const st = $("c-status");
    if (!on) {
      st.textContent = "Aucune calibration ouverte. Connecter le robot (onglet Robot), puis ouvrir.";
      $("c-robot").textContent = ""; return;
    }
    const r = CAL.robot || {};
    $("c-robot").textContent = `Robot : ${r.name || "?"}${r.naoqi_version ? " · NAOqi " + r.naoqi_version : ""}`;
    st.innerHTML = `<b>${esc(CAL.message || "")}</b>` +
      (CAL.sim ? `<div class="sim">Simulation : un démonstrateur synthétique remplace votre main et les poses de remplacement sont placées avant capture. Valeurs sans portée réelle, activation impossible.</div>` : "") +
      (CAL.error ? `<div class="err">${esc(CAL.error)}</div>` : "");
    if (CAL.sweeping) group = CAL.sweeping;
    // groupes
    const box = $("c-groups");
    if (!box.dataset.built) {
      box.innerHTML = '<span class="hint">Groupe :</span>' + Object.keys(CAL.groups).map((g) =>
        `<button data-group="${g}">${glabel(g)} <span class="done" id="c-done-${g}"></span></button>`).join("");
      box.querySelectorAll("[data-group]").forEach((b) => b.onclick = () => { group = b.dataset.group; render(); });
      box.dataset.built = "1";
    }
    for (const g of Object.keys(CAL.groups)) {
      const b = box.querySelector(`[data-group="${g}"]`);
      b.classList.toggle("sel", g === group);
      b.disabled = !!CAL.sweeping && g !== CAL.sweeping;
      const marks = [];
      if (CAL.swept.includes(g)) marks.push("exploré");
      if ((CAL.imported || {})[g]) marks.push("repris de " + CAL.imported[g].source);
      for (const p of ["rest", "start"]) if ((CAL.poses[p] || {})[g]) marks.push(POSE_LABEL[p]);
      $("c-done-" + g).textContent = marks.length ? "✓ " + marks.join(", ") : "";
    }
    // boutons
    const sweeping = !!CAL.sweeping;
    $("c-sweep").disabled = sweeping;
    $("c-sweep-stop").disabled = !sweeping;
    $("c-sweep-clear").disabled = sweeping || !CAL.swept.includes(group);
    const imported = CAL.imported || {};
    $("c-import").disabled = sweeping || !$("c-import-file").value;
    $("c-drop-import").disabled = sweeping || !(group in imported);
    $("c-discard").disabled = sweeping;
    $("c-persist").textContent = CAL.persist ? "Progression gardée à chaque étape (reprise à la prochaine ouverture)."
      : "Simulation : progression non gardée.";
    $("c-sweep-n").textContent = sweeping ? `${CAL.sweep_n} mesures…` : "";
    document.querySelectorAll("[data-cal]").forEach((b) => b.disabled = sweeping);
    const missing = CAL.missing_poses || [];
    const unswept = Object.keys(CAL.groups).filter((g) => !CAL.swept.includes(g) && !(g in (CAL.imported || {})));
    $("c-missing").textContent = (unswept.length ? `À explorer : ${unswept.map(glabel).join(", ")}. ` : "") +
      (missing.length ? `Poses manquantes : ${missing.map((m) => { const [p, g] = m.split("/"); return POSE_LABEL[p] + " " + glabel(g); }).join(", ")}.` : "");
    $("c-compute").disabled = sweeping || unswept.length > 0 || missing.length > 0;
    $("c-save").disabled = sweeping || !CAL.result || !!CAL.saved;
    $("c-activate").disabled = sweeping || !CAL.saved || CAL.activated || CAL.sim;
    $("c-saved").textContent = CAL.activated ? `${CAL.saved} active` : CAL.saved ? `${CAL.saved} enregistrée (non active)` : "";
    renderTable();
    renderPairs();
    renderResult();
    if (window.NAOFIG) window.NAOFIG.update();
  }

  function renderTable() {
    const js = CAL.groups[group] || [];
    const res = CAL.result;
    $("c-table-title").textContent = `Articulations · ${glabel(group)}`;
    const fac = CAL.factory || {};
    let h = "<tr><th>Articulation</th><th>Rôle</th><th>Actuel</th><th>Plage d'usine</th><th>Min exploré</th><th>Max exploré</th><th>Repos</th><th>Départ</th>" +
      (res ? "<th>Guardian</th>" : "") + "</tr>";
    for (const j of js) {
      const rg = CAL.ranges[j] || {};
      const rest = ((CAL.poses.rest || {})[group] || {})[j], start = ((CAL.poses.start || {})[group] || {})[j];
      const role = (CAL.roles || {})[j] || "—";
      h += `<tr><td>${j}</td><td>${role === "guided" ? "relâché" : role === "driven" ? "raidi" : role}</td>` +
        `<td class="live num">${CAL.angles ? f3(CAL.angles[j]) : "—"}</td>` +
        `<td class="num">${fac[j] ? f3(fac[j][0]) + " … " + f3(fac[j][1]) + "<br>" + facBar(fac[j], CAL.angles ? CAL.angles[j] : null, rg) : "non lues"}</td>` +
        `<td class="num${rg.live ? " live" : ""}">${f3(rg.min)}</td><td class="num${rg.live ? " live" : ""}">${f3(rg.max)}</td>` +
        `<td class="num">${f3(rest)}</td><td class="num">${f3(start)}</td>` +
        (res ? `<td class="num">${res.limits[j] ? f3(res.limits[j][0]) + " … " + f3(res.limits[j][1]) : "—"}</td>` : "") + "</tr>";
    }
    $("c-table").innerHTML = h;
  }

  // barre : plage d'usine (fond), plage explorée (bande), angle actuel (trait) ;
  // le trait passe en rouge à moins de 0,05 rad d'une limite d'usine
  function facBar(f, q, rg) {
    if (!f) return "—";
    const W = 110, H = 10, X = (v) => Math.max(0, Math.min(W, (v - f[0]) / ((f[1] - f[0]) || 1) * W));
    const near = q != null && (q - f[0] < 0.05 || f[1] - q < 0.05);
    const band = rg && rg.min != null ? `<rect x="${X(rg.min).toFixed(1)}" y="2" width="${Math.max(1, X(rg.max) - X(rg.min)).toFixed(1)}" height="${H - 4}" fill="var(--accent)" fill-opacity=".35"/>` : "";
    const mark = q != null ? `<rect x="${(X(q) - 1.5).toFixed(1)}" y="0" width="3" height="${H}" fill="${near ? "var(--bad, #c0392b)" : "currentColor"}"/>` : "";
    return `<svg class="facbar" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="plage d'usine">` +
      `<rect x="0" y="0" width="${W}" height="${H}" rx="2" fill="none" stroke="var(--muted)"/>${band}${mark}</svg>` +
      (near ? ' <span class="warn">près de la butée</span>' : "");
  }

  function renderPairs() {
    const S = 220, P = 22;
    $("c-pairs").innerHTML = (CAL.pairs || []).filter((p) => (CAL.groups[group] || []).includes(p.joints[0])).map((p) => {
      const [la, lb] = p.limits;
      const extra = Object.values(p.poses || {}).map((o) => o.q).concat(p.current ? [p.current] : []);
      const xs = p.points.map((q) => q[0]).concat(la, extra.map((q) => q[0])),
        ys = p.points.map((q) => q[1]).concat(lb, extra.map((q) => q[1]));
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const X = (v) => P + (v - x0) / ((x1 - x0) || 1) * (S - 2 * P), Y = (v) => S - P - (v - y0) / ((y1 - y0) || 1) * (S - 2 * P);
      const dots = p.points.map((q) => `<circle cx="${X(q[0]).toFixed(1)}" cy="${Y(q[1]).toFixed(1)}" r="1.6" fill="var(--accent)" fill-opacity=".5"/>`).join("");
      const hull = p.hull.length >= 3 ? `<polygon points="${p.hull.map((q) => X(q[0]).toFixed(1) + "," + Y(q[1]).toFixed(1)).join(" ")}" fill="var(--accent)" fill-opacity=".08" stroke="var(--accent)" stroke-width="1.5"/>` : "";
      const zone = (p.zone || []).length >= 3 ? `<polygon points="${p.zone.map((q) => X(q[0]).toFixed(1) + "," + Y(q[1]).toFixed(1)).join(" ")}" fill="var(--ok)" fill-opacity=".12" stroke="var(--ok)" stroke-width="1.5" stroke-dasharray="5 3"/>` : "";
      const ps = p.poses || {};
      const marks = (ps.rest ? `<rect x="${(X(ps.rest.q[0]) - 5).toFixed(1)}" y="${(Y(ps.rest.q[1]) - 5).toFixed(1)}" width="10" height="10" fill="var(--muted)" stroke="var(--card)"><title>repos (peut être hors zone)</title></rect>` : "") +
        (ps.start ? `<circle cx="${X(ps.start.q[0]).toFixed(1)}" cy="${Y(ps.start.q[1]).toFixed(1)}" r="6" fill="${ps.start.inside ? "var(--ok)" : "var(--bad)"}" stroke="var(--card)" stroke-width="1.5"><title>départ</title></circle>` : "") +
        (p.current ? `<g stroke="${p.current_inside ? "currentColor" : "var(--bad)"}" stroke-width="1.5"><line x1="${X(p.current[0]) - 7}" y1="${Y(p.current[1])}" x2="${X(p.current[0]) + 7}" y2="${Y(p.current[1])}"/><line x1="${X(p.current[0])}" y1="${Y(p.current[1]) - 7}" x2="${X(p.current[0])}" y2="${Y(p.current[1]) + 7}"/><title>position actuelle</title></g>` : "");
      const verdict = ps.start ? (ps.start.inside ? '<span class="okc">départ dans la zone ✓</span>'
        : '<span class="warn">départ HORS zone : explorer autour du point rouge</span>') : "";
      const frame = `<rect x="${X(la[0])}" y="${Y(lb[1])}" width="${X(la[1]) - X(la[0])}" height="${Y(lb[0]) - Y(lb[1])}" fill="none" stroke="var(--muted)" stroke-dasharray="4 3"/>`;
      return `<figure><svg viewBox="0 0 ${S} ${S}">${frame}${hull}${zone}${dots}${marks}` +
        `<text x="${S / 2}" y="${S - 5}" text-anchor="middle" font-size="11" fill="currentColor">${p.joints[0]}</text>` +
        `<text x="10" y="${S / 2}" font-size="11" fill="currentColor" transform="rotate(-90 10 ${S / 2})" text-anchor="middle">${p.joints[1]}</text></svg>` +
        `<figcaption>${p.n} mesures ${verdict}</figcaption></figure>`;
    }).join("") || '<span class="hint">Aucun couple exploré pour ce groupe.</span>';
  }

  function renderResult() {
    const res = CAL.result, box = $("c-result");
    box.classList.toggle("hidden", !res);
    if (!res) return;
    let h = `<h3>Calibration ${esc(res.version)} : ${Object.keys(res.limits).length} articulations, ${res.coupled} contraintes couplées, marge ${res.margin_rad} rad</h3>` +
      ((res.warnings || []).length ? `<div class="banner-warn"><b>Attention :</b> l'exploration a dépassé des limites d'usine ; ce sont les limites d'usine qui sont gardées.<ul>${res.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>` : "") +
      "<table><tr><th>Articulation</th><th>Mesuré</th><th>Usine</th><th>Guardian</th></tr>";
    for (const [j, lim] of Object.entries(res.limits)) {
      const m = res.measured[j] || {};
      h += `<tr${m.beyond_factory ? ' class="beyond"' : ""}><td>${j}${m.beyond_factory ? ' <span class="warn">au-delà de l\'usine</span>' : ""}</td><td class="num">${f3(m.min)} … ${f3(m.max)}</td>` +
        `<td class="num">${m.factory ? f3(m.factory[0]) + " … " + f3(m.factory[1]) : "non lues"}</td>` +
        `<td class="num">${f3(lim[0])} … ${f3(lim[1])}</td></tr>`;
    }
    box.innerHTML = h + "</table>";
  }

  // -- calibrations enregistrées (liste + activation, même sans calibration ouverte) --
  let FILES = [];
  async function loadFiles() {
    let d;
    try { d = await (await fetch("/api/calibrations")).json(); } catch (e) { return; }
    if (!d || !d.files) return;
    FILES = d.files;
    const box = $("c-files");
    if (!FILES.length) {
      box.textContent = "Aucun fichier dans configs/calibration/.";
    } else {
      box.innerHTML = "<table><tr><th>Fichier</th><th>Robot</th><th>Créée</th><th>État</th><th></th></tr>" +
        FILES.map((f) => `<tr class="${f.active ? "is-active" : ""}"><td>${esc(f.file)}${f.notes ? `<div class="hint">${esc(f.notes)}</div>` : ""}</td>` +
          `<td>${esc((f.robot || {}).name || "?")}${f.simulated ? " (simulation)" : ""}</td><td>${esc(f.created_at || "")}</td>` +
          `<td>${f.active ? "active" : f.activable ? "activable" : `<span class="why">refusée : ${esc(f.reason || "")}</span>`}</td>` +
          `<td>${f.active || !f.activable ? "" : `<button data-activate="${esc(f.file)}" class="ok">Activer</button>`}</td></tr>`).join("") +
        "</table>";
      box.querySelectorAll("[data-activate]").forEach((b) => b.onclick = async () => {
        b.disabled = true;
        try { await api("calibrations/activate", {file: b.dataset.activate}); } catch (e) { /* affiché */ }
        await loadFiles();
        if (typeof refreshStatus === "function") refreshStatus();
      });
    }
    const sel = $("c-import-file"), prev = sel.value;
    const usable = FILES.filter((f) => f.valid && !f.simulated);
    sel.innerHTML = usable.length ? usable.map((f) => `<option value="${esc(f.file)}">${esc(f.version || f.file)}</option>`).join("")
      : '<option value="">aucun fichier réel</option>';
    if (usable.some((f) => f.file === prev)) sel.value = prev;
  }

  async function poll() {
    if (busy || !$("tab-calib").classList.contains("active")) return;
    try { CAL = await (await fetch("/api/calibration")).json(); render(); } catch (e) { /* hors ligne */ }
  }

  $("c-open").onclick = () => call("open");
  $("c-close").onclick = () => window.confirm("Fermer sans enregistrer ? Les articulations restent dans leur état actuel.") && call("close");
  $("c-sweep").onclick = () => confirmRelease() && call("sweep_start", {group, confirm: true});
  $("c-sweep-stop").onclick = () => call("sweep_stop");
  $("c-sweep-clear").onclick = () => window.confirm(`Effacer les mesures d'exploration du groupe « ${glabel(group)} » ?`) && call("sweep_clear", {group});
  document.querySelectorAll("[data-cal]").forEach((b) => b.onclick = () => {
    const op = b.dataset.cal;
    if (op === "release") return confirmRelease() && call("release", {group, confirm: true});
    if (op === "hold") return call("hold", {group});
    return call("capture", {group, pose: b.dataset.pose});
  });
  $("c-compute").onclick = () => call("compute", {version: $("c-version").value.trim()});
  $("c-save").onclick = () => call("save").then(loadFiles);
  $("c-activate").onclick = () => window.confirm("Activer cette calibration pour le guardian ? Le contrôleur reste désarmé.") &&
    call("activate", {confirm: true}).then(loadFiles);
  $("c-import").onclick = () => call("import", {file: $("c-import-file").value, group});
  $("c-drop-import").onclick = () => call("drop_import", {group});
  $("c-discard").onclick = () => window.confirm("Effacer toute la progression de cette calibration (explorations, poses, reprises) ?") &&
    call("discard");
  loadFiles();
  document.querySelector('nav button[data-tab="calib"]').addEventListener("click", loadFiles);
  setInterval(poll, 300);
  document.querySelector('nav button[data-tab="calib"]').addEventListener("click", poll);
  // lu par nao-figure.js (illustration) : groupe affiché et état de la calibration ;
  // un clic sur une partie du robot choisit le groupe, comme les boutons
  window.CALIB = {
    get active() { return !!CAL.active; },
    get group() { return group; },
    get groupLabel() { return glabel(group); },
    // raccourci Espace (shortcuts.js) : relâche le groupe choisi, comme le bouton, sans boîte de dialogue
    release() { return CAL.active ? call("release", {group, confirm: true}) : undefined; },
    get sweeping() { return CAL.sweeping || null; },
    select(g) { if (CAL.active && !CAL.sweeping && CAL.groups && g in CAL.groups) { group = g; render(); } },
  };
})();
