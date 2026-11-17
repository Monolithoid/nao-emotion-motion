"use strict";
// Network tab (MLP settings + live diagram inputs -> hidden layers -> outputs) and Training tab
// (runs in chronological order on the left, charts of the selected run on the right, with the
// values to expect). Nothing here moves the robot. Training settings are sent as `overrides`
// with POST /api/train (configs/model.yaml is never modified).
(function () {
  const GROUPS = {arms: "Bras", head: "Tête", hands: "Mains"};
  const GCOL = {arms: "#2f62d8", head: "#e07a12", hands: "#1d8a4b"};
  const FIELDS = {hidden: "tr-hidden", dropout: "tr-dropout", weight_decay: "tr-wd", lr: "tr-lr", smoothness_weight: "tr-smooth", folds: "tr-folds", n_control_points: "tr-ctrl"};
  const STORE = "nao-net-settings";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const fr = (x, d = 3) => (x === undefined || x === null || !isFinite(x)) ? "—" : Number(x).toFixed(d).replace(".", ",");
  const pct = (x) => (x >= 0 ? "−" : "+") + Math.abs(Math.round(x * 100)) + " %";
  let netGroup = "arms", runs = [], selected = null, filter = "", newestFirst = false, hist = {}, lastSig = "", progress = null;
  const runSel = new Set(), runTrashSel = new Set();  // ticked trainings / ticked trashed trainings
  let runTrash = [];

  // ---- settings -----------------------------------------------------------------------
  const M = () => (CFG.model || {});
  const groupModel = (g) => M()[{arms: "arm_model", head: "head_model", hands: "hands_model"}[g]] || M().head_model || {};
  function overrides() {
    const o = {};
    for (const [k, id] of Object.entries(FIELDS)) if ($(id).value.trim() !== "") o[k] = $(id).value.trim();
    return o;
  }
  const mirrorDefault = () => (M().training || {}).mirror ?? true;
  const foldsSetting = () => { const v = $("tr-folds").value.trim(); return parseInt(v !== "" ? v : ((M().training || {}).folds ?? 1)) || 0; };
  function setFields(v) { for (const [k, id] of Object.entries(FIELDS)) $(id).value = v[k] ?? ""; changed(); }
  function placeholders() {
    const A = groupModel(netGroup), T = M().training || {};
    $("tr-hidden").placeholder = (A.hidden || []).join(","); $("tr-dropout").placeholder = A.dropout ?? "";
    $("tr-wd").placeholder = T.weight_decay ?? ""; $("tr-lr").placeholder = T.lr ?? ""; $("tr-smooth").placeholder = T.smoothness_weight ?? "";
    $("tr-folds").placeholder = T.folds ?? 1;
    $("tr-ctrl").placeholder = (M().representation || {}).n_control_points ?? 8;
  }
  // effective settings for a group: field if filled, else model.yaml
  function effective(g) {
    const A = groupModel(g), T = M().training || {}, o = overrides();
    let hidden = (A.hidden || [64, 64]).slice();
    if (o.hidden !== undefined) {
      const h = o.hidden.replace(/,/g, " ").split(/\s+/).filter(Boolean).map(Number);
      hidden = h.length && h.every((x) => Number.isInteger(x) && x >= 1 && x <= 1024) && h.length <= 4 ? h : null;
    }
    const num = (k, d) => (o[k] !== undefined ? Number(o[k]) : d);
    return {hidden, dropout: num("dropout", A.dropout || 0), weight_decay: num("weight_decay", T.weight_decay),
            lr: num("lr", T.lr), smoothness_weight: num("smoothness_weight", T.smoothness_weight),
            n_control_points: num("n_control_points", (M().representation || {}).n_control_points || 8)};
  }
  function changed() {
    try { localStorage.setItem(STORE, JSON.stringify({...overrides(), mirror: $("tr-mirror").checked})); } catch (e) {}
    drawNet(); summary();
  }
  function summary() {
    const s = effective("arms"), o = overrides();
    $("tr-net-sum").textContent = Object.keys(o).length
      ? `Réglages de l'onglet Réseau : ${Object.entries(o).map(([k, v]) => `${{hidden: "couches", dropout: "dropout", weight_decay: "weight decay", lr: "lr", smoothness_weight: "lissage", folds: "parts de validation croisée", n_control_points: "points de contrôle"}[k]} ${v}`).join(" · ")}`
      : "Réglages : ceux de model.yaml (modifiables dans l'onglet Réseau)";
    $("tr-net-sum").textContent += $("tr-mirror").checked ? " · miroir activé" : " · miroir désactivé";
    const k = foldsSetting();
    $("tr-net-sum").textContent += k >= 2 ? ` · validation croisée ${k} parts (≈ ${k + 1} entraînements)` : " · découpage unique 75/25";
    return s;
  }

  // ---- network diagram ----------------------------------------------------------------
  const inDim = () => CFG.emotions.length + 1 + CFG.net.event_features.length;
  // control points per joint: the Network tab field if valid, else model.yaml
  const K = () => { const k = effective(netGroup).n_control_points; return Number.isInteger(k) && k >= 4 && k <= 64 ? k : ((M().representation || {}).n_control_points || 8); };
  const joints = (g) => (CFG.net.joints[g] || []);
  function nParams(dims) { let n = 0; for (let i = 1; i < dims.length; i++) n += dims[i - 1] * dims[i] + dims[i]; return n; }

  function drawNet() {
    const svg = $("net-svg"); if (!svg) return;
    const W = 1200, H = 660, top = 54, bot = 600;
    const s = effective(netGroup), J = joints(netGroup), k = K();
    if (!s.hidden) { svg.innerHTML = `<text x="600" y="330" text-anchor="middle" class="nt-bad">Couches cachées invalides : 1 à 4 nombres entre 1 et 1024 (ex. 32,32)</text>`; stats(null); return; }
    const outDim = k * J.length + 1, dims = [inDim(), ...s.hidden, outDim];
    // columns: inputs, hidden layers, outputs, decoder
    const xIn = 330, xOut = 900, xDec = 1080, nh = s.hidden.length;
    const xs = [xIn, ...s.hidden.map((_, i) => xIn + (xOut - xIn) * (i + 1) / (nh + 1)), xOut];
    const out = [];
    const ys = (n, gap = 1) => { const step = Math.min(40, (bot - top) / Math.max(n, 1)); const h = step * (n - 1); return Array.from({length: n}, (_, i) => (top + bot) / 2 - h / 2 + i * step * gap); };
    // input nodes (labelled), grouped
    const ev = CFG.net.event_features;
    const inLab = [...CFG.emotions.map((e) => ({t: e.label, g: "emo"})), {t: "intensité", g: "int"}, ...ev.map((t) => ({t, g: "ev"}))];
    const inY = (() => { const n = inLab.length, step = Math.min(38, (bot - top - 40) / n); let y = (top + bot) / 2 - (step * (n - 1) + 40) / 2, prev = null; return inLab.map((l) => { if (prev && prev !== l.g) y += 20; prev = l.g; const r = y; y += step; return r; }); })();
    // hidden nodes: at most 12 drawn per layer, "⋮" in the middle
    const hidNodes = s.hidden.map((n) => { const shown = Math.min(n, 12); const yy = ys(shown + (n > 12 ? 1 : 0)); return n > 12 ? {y: [...yy.slice(0, 6), ...yy.slice(7)], dots: yy[6], n} : {y: yy, n}; });
    // outputs: one box per joint (k control points) + duration
    const outRows = [...J.map((j) => ({t: j, sub: `${k} pts`})), {t: "log durée", sub: "1"}];
    const oStep = Math.min(44, (bot - top) / outRows.length), oH = Math.min(34, oStep - 6);
    const oY = outRows.map((_, i) => (top + bot) / 2 - oStep * (outRows.length - 1) / 2 + i * oStep);
    const cols = [inY, ...hidNodes.map((h) => h.y), oY];
    // connections (between drawn nodes only)
    for (let c = 0; c < cols.length - 1; c++) {
      const x0 = xs[c] + (c === 0 ? 8 : 9), x1 = xs[c + 1] - (c + 1 === cols.length - 1 ? 70 : 9);
      const many = cols[c].length * cols[c + 1].length;
      const op = Math.max(0.05, Math.min(0.35, 18 / many));
      out.push(`<g class="nt-edge" style="stroke-opacity:${op}">`);
      for (const a of cols[c]) for (const b of cols[c + 1]) out.push(`<line x1="${x0}" y1="${a}" x2="${x1}" y2="${b}"/>`);
      out.push("</g>");
    }
    // input column
    const gl = {emo: "émotion (one-hot)", int: "intensité 0..1", ev: "événement"};
    const gcol = {emo: "var(--accent)", int: "var(--guided)", ev: "var(--ok)"};
    let prev = null;
    inLab.forEach((l, i) => {
      if (l.g !== prev) {
        const idx = inLab.map((x, j) => x.g === l.g ? j : -1).filter((j) => j >= 0);
        const y0 = inY[idx[0]] - 12, y1 = inY[idx[idx.length - 1]] + 12;
        out.push(`<path d="M${xIn - 128} ${y0} h-6 V${y1} h6" class="nt-brace" style="stroke:${gcol[l.g]}"/>`,
          `<text x="${xIn - 142}" y="${(y0 + y1) / 2}" class="nt-group" text-anchor="end" dominant-baseline="middle" style="fill:${gcol[l.g]}">${esc(gl[l.g])}</text>`);
        prev = l.g;
      }
      out.push(`<circle cx="${xIn}" cy="${inY[i]}" r="8" class="nt-node" style="stroke:${gcol[l.g]}"/>`,
        `<text x="${xIn - 16}" y="${inY[i]}" class="nt-lab" text-anchor="end" dominant-baseline="middle">${esc(l.t)}</text>`);
    });
    out.push(`<text x="${xIn}" y="22" class="nt-head" text-anchor="middle">Entrées · ${dims[0]}</text>`);
    // hidden columns
    hidNodes.forEach((h, i) => {
      const x = xs[i + 1];
      const yA = Math.min(...h.y) - 20, yB = Math.max(...h.y) + 20;
      out.push(`<rect x="${x - 26}" y="${yA}" width="52" height="${yB - yA}" rx="14" class="nt-layer"/>`);
      h.y.forEach((y) => out.push(`<circle cx="${x}" cy="${y}" r="9" class="nt-node nt-hid"/>`));
      if (h.dots !== undefined) out.push(`<text x="${x}" y="${h.dots + 6}" class="nt-dots" text-anchor="middle">⋮</text>`);
      out.push(`<text x="${x}" y="22" class="nt-head" text-anchor="middle">Couche ${i + 1} · ${h.n}</text>`,
        `<text x="${x}" y="40" class="nt-sub" text-anchor="middle">tanh${s.dropout > 0 ? " · dropout " + fr(s.dropout, 2) : ""}</text>`);
    });
    // outputs
    out.push(`<text x="${xOut}" y="22" class="nt-head" text-anchor="middle">Sorties · ${outDim}</text>`,
      `<text x="${xOut}" y="40" class="nt-sub" text-anchor="middle">linéaire</text>`);
    outRows.forEach((r, i) => {
      const dur = i === outRows.length - 1;
      out.push(`<rect x="${xOut - 70}" y="${oY[i] - oH / 2}" width="140" height="${oH}" rx="6" class="nt-out${dur ? " dur" : ""}"/>`,
        `<text x="${xOut}" y="${oY[i] - (oH > 26 ? 4 : 0)}" class="nt-olab" text-anchor="middle" dominant-baseline="middle">${esc(r.t)}</text>`);
      if (oH > 26) out.push(`<text x="${xOut}" y="${oY[i] + 10}" class="nt-sub" text-anchor="middle" dominant-baseline="middle">${r.sub}</text>`);
    });
    // decoder
    const dy0 = (top + bot) / 2 - 120;
    out.push(`<path d="M${xOut + 74} ${(top + bot) / 2} H${xDec - 84}" class="nt-arrow" marker-end="url(#nt-ar)"/>`,
      `<defs><marker id="nt-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" class="nt-arhead"/></marker></defs>`,
      `<rect x="${xDec - 80}" y="${dy0}" width="200" height="240" rx="10" class="nt-dec"/>`,
      `<text x="${xDec + 20}" y="${dy0 + 28}" class="nt-head" text-anchor="middle">Décodage</text>`);
    const dl = [`B-spline cubique`, `${k} points / articulation`, `+ durée (exp)`, `→ trajectoire de ${J.length}`, `articulation${J.length > 1 ? "s" : ""} (${netGroup === "hands" ? "ouverture" : "rad"})`, ``, `guardian : au jeu`, `seulement`];
    dl.forEach((t, i) => out.push(`<text x="${xDec + 20}" y="${dy0 + 58 + i * 22}" class="nt-sub" text-anchor="middle">${esc(t)}</text>`));
    out.push(`<text x="600" y="${H - 16}" class="nt-sub" text-anchor="middle">Chaque ligne = une connexion pondérée (seuls 12 neurones par couche sont dessinés). Les trois groupes ont chacun leur réseau, mêmes entrées.</text>`);
    svg.innerHTML = out.join("");
    stats(dims);
  }

  function stats(dims) {
    const el = $("net-stats"); if (!el) return;
    if (!dims) { el.innerHTML = ""; return; }
    const P = nParams(dims), g = netGroup;
    const acc = progress && progress.kinds && progress.kinds[g] ? progress.kinds[g].accepted : null;
    const targets = acc === null ? null : acc * (dims[dims.length - 1]);
    const ratio = targets ? P / targets : null;
    const cls = ratio === null ? "" : ratio <= 1 ? "v-good" : ratio <= 3 ? "v-warn" : "v-bad";
    el.innerHTML = `<div><b>${P.toLocaleString("fr-FR")}</b> paramètres (${dims.join(" → ")})</div>` +
      (acc === null ? "" : `<div>${acc} prise${acc > 1 ? "s" : ""} acceptée${acc > 1 ? "s" : ""} (${GROUPS[g].toLowerCase()}) × ${dims[dims.length - 1]} valeurs = ${targets.toLocaleString("fr-FR")} valeurs à apprendre</div>` +
        `<div class="verdict ${cls}">${ratio === null ? "" : `${fr(ratio, 1)} paramètre${ratio >= 2 ? "s" : ""} par valeur apprise · ` +
          (cls === "v-good" ? "taille raisonnable pour ces données" : cls === "v-warn" ? "réseau un peu gros pour ces données : surveiller l'écart entraînement / validation" : "réseau très gros pour ces données : risque de surapprentissage")}</div>` +
        `<div class="legend">Repère indicatif : ≤ 1 paramètre par valeur apprise (vert), ≤ 3 (orange). Le verdict réel est la validation dans l'onglet Entraînement.</div>`);
  }

  // ---- how many control points: fidelity of the curves on the takes ------------------------
  const fid = {};
  function measureFidelity() {
    const g = netGroup;
    $("net-fid").innerHTML = `<p class="hint">Mesure en cours…</p>`;
    api("fidelity?component=" + g, undefined, {quiet: true})
      .then((d) => { fid[g] = d; if (netGroup === g) fidelityView(); })
      .catch((e) => { if (netGroup === g) $("net-fid").innerHTML = `<p class="hint">${esc(e.message)}</p>`; });
  }
  function fidelityView() {
    const el = $("net-fid"), d = fid[netGroup]; if (!el) return;
    if (!d) { el.innerHTML = ""; return; }
    if (!d.rows.length) { el.innerHTML = `<p class="hint">Aucune prise acceptée pour ce groupe.</p>`; return; }
    const k = K();
    el.innerHTML = `<table class="fid"><thead><tr><th>Points</th><th title="Écart moyen entre les prises et leur courbe (toutes articulations)">Écart moyen</th><th title="La prise la plus mal reproduite">Pire prise</th><th title="Part médiane de l'amplitude des articulations qui bougent (plus de 0,1 rad) gardée par la courbe ; entre parenthèses, la pire">Amplitude gardée</th></tr></thead><tbody>` +
      d.rows.map((r) => { const a = r.amplitude_kept, cls = r.rmse_rad <= 0.02 ? "v-good" : r.rmse_rad <= 0.05 ? "v-warn" : "v-bad";
        return `<tr data-k="${r.n_control_points}" class="${r.n_control_points === k ? "on" : ""}" title="Utiliser ${r.n_control_points} points pour les prochains entraînements"><td>${r.n_control_points}</td><td class="${cls}">${fr(r.rmse_rad)} rad</td><td>${fr(r.worst_take_rmse_rad)} rad</td><td>${a === null ? "—" : Math.round(a * 100) + " %"}${r.amplitude_kept_min !== null ? ` (${Math.round(r.amplitude_kept_min * 100)} %)` : ""}</td></tr>`; }).join("") +
      `</tbody></table><div class="legend">${d.n_takes} prise${d.n_takes > 1 ? "s" : ""} acceptée${d.n_takes > 1 ? "s" : ""}${d.real_takes ? "" : " SIMULÉES (aucune prise réelle)"}. Ce que la courbe perd ici, aucun modèle ne peut le produire. Écart ≤ 0,02 rad (≈ 1°) : assez de points, un geste généré trop lisse vient alors du modèle (il fait la moyenne des prises). Cliquer une ligne pour choisir ce nombre de points.</div>`;
  }

  // ---- training runs ------------------------------------------------------------------
  const fmtStamp = (s) => { const m = /^(\d{4})(\d\d)(\d\d)-(\d\d)(\d\d)(\d\d)/.exec(s || ""); return m ? `${m[3]}/${m[2]} ${m[4]}:${m[5]}` : s; };
  // what to expect: returns [{k, cls, text}] for a run's metrics
  function assess(m) {
    const out = [], v = m.val_rmse_rad, b = m.val_baseline_label_mean_rmse_rad, t = m.train_rmse_rad;
    const cv = m.cv;
    if (v === undefined) { out.push({k: "val", cls: "v-bad", text: "pas de validation (moins de deux groupes de prises)"}); }
    else {
      const gain = 1 - v / b;
      out.push(gain >= 0.1 ? {k: "base", cls: "v-good", text: `validation ${pct(gain)} vs geste moyen : le réseau apprend mieux que la moyenne`}
        : gain >= 0 ? {k: "base", cls: "v-warn", text: `validation ${pct(gain)} vs geste moyen : à peine mieux, sous-apprentissage probable`}
        : {k: "base", cls: "v-bad", text: `validation ${pct(gain)} vs geste moyen : moins bien que la moyenne`});
      const gap = v / Math.max(t, 1e-9);
      out.push(gap <= 1.5 ? {k: "gap", cls: "v-good", text: `écart validation / entraînement ×${fr(gap, 1)} : raisonnable`}
        : gap <= 2.5 ? {k: "gap", cls: "v-warn", text: `écart validation / entraînement ×${fr(gap, 1)} : début de surapprentissage`}
        : {k: "gap", cls: "v-bad", text: `écart validation / entraînement ×${fr(gap, 1)} : surapprentissage (colle aux prises d'entraînement)`});
    }
    if (cv && cv.ratio_mean !== null && cv.ratio_mean !== undefined) {
      const sd = cv.ratio_std, cls = sd <= 0.1 ? "v-good" : sd <= 0.2 ? "v-warn" : "v-bad";
      out.push({k: "cv", cls, text: `validation croisée ${cv.folds} parts : ${pct(1 - cv.ratio_mean)} ± ${Math.round(sd * 100)} % vs geste moyen selon la part` +
        (cls === "v-good" ? " : résultat stable" : cls === "v-warn" ? " : résultat assez variable d'une part à l'autre" : " : résultat très variable, il manque des prises")});
      out.push({k: "kept", cls: "v-good", text: `modèle gardé : réentraîné sur les ${m.n_examples} prises pendant ${cv.final_epochs} époques (meilleure époque moyenne des parts, ${Math.round(cv.best_epoch_mean)} ± ${Math.round(cv.best_epoch_std)})`});
    }
    const st = m.settings || {};
    if (m.best_epoch !== undefined && st.epochs) {
      const r = m.best_epoch / st.epochs;
      if (r >= 0.9) out.push({k: "ep", cls: "v-warn", text: `meilleure validation en fin d'entraînement (époque ${m.best_epoch}) : allonger l'entraînement`});
      else if (r <= 0.05) out.push({k: "ep", cls: "v-warn", text: `meilleure validation très tôt (époque ${m.best_epoch}) : réseau trop gros ou lr trop fort`});
      else out.push({k: "ep", cls: "v-good", text: `meilleure validation à l'époque ${m.best_epoch} / ${st.epochs}`});
    }
    if (cv) { const small = Math.min(...cv.per_fold.map((f) => f.n_val)); if (small < 5) out.push({k: "nval", cls: "v-warn", text: `parts de ${small} prises seulement : moins de parts donnerait des notes moins bruitées`}); }
    else if (m.n_val !== undefined && m.n_val < 8) out.push({k: "nval", cls: "v-warn", text: `validation sur ${m.n_val} prises seulement : chiffres bruités`});
    if (m.guardian && m.guardian.total) {
      const r = m.guardian.accepted / m.guardian.total;
      out.push({k: "gd", cls: r >= 0.9 ? "v-good" : r >= 0.5 ? "v-warn" : "v-bad", text: `guardian : ${m.guardian.accepted}/${m.guardian.total} gestes générés jouables (au jeu)`});
    }
    return out;
  }
  const worst = (a) => a.some((x) => x.cls === "v-bad") ? "v-bad" : a.some((x) => x.cls === "v-warn") ? "v-warn" : "v-good";

  function renderList() {
    const el = $("tr-runs");
    let list = runs.map((r, i) => ({r, i})).filter((x) => !filter || x.r.component === filter);
    if (newestFirst) list = list.slice().reverse();
    if (!list.length) { el.innerHTML = `<p class="hint">Aucun entraînement${filter ? " pour ce groupe" : ""}.</p>`; return; }
    el.innerHTML = `<table class="tr-table"><thead><tr><th><input type="checkbox" id="tr-all" title="Tick every training shown"></th><th>#</th><th>Date</th><th>Groupe</th><th>Réseau</th><th class="num">Valid.</th><th class="num">Réf.</th><th class="num">vs réf.</th><th class="num">Guardian</th></tr></thead><tbody>` +
      list.map(({r, i}) => {
        const m = r.metrics || {}, s = m.settings || {}, v = m.val_rmse_rad, b = m.val_baseline_label_mean_rmse_rad;
        const gain = v !== undefined && b ? 1 - v / b : null, a = assess(m);
        return `<tr class="clickable${r.run === selected ? " on" : ""}${runSel.has(r.run) ? " ticked" : ""}" data-run="${esc(r.run)}" title="${esc(r.run)}">` +
          `<td class="sel"><input type="checkbox" data-rtick="${esc(r.run)}"${runSel.has(r.run) ? " checked" : ""}></td>` +
          `<td>${i + 1}</td><td>${fmtStamp(r.stamp)}</td><td><span class="dot" style="background:${GCOL[r.component] || "#888"}"></span>${GROUPS[r.component] || esc(r.component)}${r.active ? ` <span class="tag active" title="Active model of this group: used by generation and published by export-dataset">active</span>` : ""}</td>` +
          `<td>${s.hidden ? s.hidden.join("-") + (s.dropout ? ` · d${fr(s.dropout, 2)}` : "") + (s.mirror ? " · ⇆" : "") + (s.folds > 1 ? ` · ×${s.folds}` : "") : "—"}</td>` +
          `<td class="num">${fr(v)}</td><td class="num">${fr(b)}</td>` +
          `<td class="num"><span class="verdict ${gain === null ? "" : gain >= 0.1 ? "v-good" : gain >= 0 ? "v-warn" : "v-bad"}">${gain === null ? "—" : pct(gain)}</span></td>` +
          `<td class="num">${m.guardian ? m.guardian.accepted + "/" + m.guardian.total : "—"} <span class="vdot ${worst(a)}" title="${esc(a.map((x) => x.text).join("\n"))}"></span></td></tr>`;
      }).join("") + "</tbody></table>";
    runSelInfo();
    const on = el.querySelector("tr.on");
    if (on && el.dataset.scrolled !== selected) { on.scrollIntoView({block: "nearest"}); el.dataset.scrolled = selected; }
  }

  // ---- selection and trash of trainings ------------------------------------------------
  const visibleRuns = () => runs.filter((r) => !filter || r.component === filter);
  function runSelInfo() {
    const n = runSel.size, act = runs.filter((r) => runSel.has(r.run) && r.active);
    $("tr-trash").disabled = !n;
    $("tr-trash").textContent = n ? `Move ${n} training${n > 1 ? "s" : ""} to trash` : "Move to trash";
    $("tr-sel-info").innerHTML = act.length
      ? `<span class="v-warn">includes the active model of ${act.map((r) => GROUPS[r.component]).join(", ")}: the previous training of that group becomes the active model</span>`
      : (n ? `${n} ticked` : "");
    const all = $("tr-all"), vis = visibleRuns();
    if (all) { all.checked = n > 0 && vis.every((r) => runSel.has(r.run)); all.indeterminate = n > 0 && !all.checked; }
  }
  async function loadRunTrash() {
    const d = await api("trash", undefined, {quiet: true}).catch(() => null);
    if (!d) return;
    runTrash = d.runs;
    for (const r of [...runTrashSel]) if (!runTrash.some((x) => x.run === r)) runTrashSel.delete(r);
    $("tr-trash-sum").innerHTML = `Trash: <b>${runTrash.length}</b> training${runTrash.length > 1 ? "s" : ""}` +
      (runTrash.length ? ` (${fmtBytes(d.runs_bytes)}, in <code>runs/trash/</code>, restorable)` : " (empty)");
    $("tr-empty").disabled = !runTrash.length;
    $("tr-trash-list").innerHTML = runTrash.length ? `<table class="tr-table"><thead><tr><th><input type="checkbox" id="trt-all"></th><th>Date</th><th>Groupe</th><th>Size</th><th>Trashed</th></tr></thead><tbody>` +
      runTrash.map((r) => `<tr><td class="sel"><input type="checkbox" data-rttick="${esc(r.run)}"${runTrashSel.has(r.run) ? " checked" : ""}></td><td>${fmtStamp(r.stamp)}</td>` +
        `<td><span class="dot" style="background:${GCOL[r.component] || "#888"}"></span>${GROUPS[r.component] || esc(r.component)}${r.was_active ? ` <span class="tag" title="Was the active model when trashed">was active</span>` : ""}</td>` +
        `<td>${fmtBytes(r.bytes)}</td><td>${esc((r.trashed_at || "").replace("T", " "))}</td></tr>`).join("") + "</tbody></table>" : "";
    runTrashInfo();
  }
  function runTrashInfo() {
    const n = runTrashSel.size;
    $("tr-restore").disabled = $("tr-purge").disabled = !n;
    $("tr-restore").textContent = n ? `Restore ${n}` : "Restore selected";
    $("tr-purge").textContent = n ? `Delete ${n} for good` : "Delete selected for good";
    const all = $("trt-all");
    if (all) { all.checked = n > 0 && n === runTrash.length; all.indeterminate = n > 0 && !all.checked; }
  }
  const activeMsg = (ch) => Object.entries(ch || {}).map(([g, r]) => `${GROUPS[g]}: ${r ? "active model is now the training of " + fmtStamp(r.split("-").slice(1).join("-")) : "no model left"}`).join(" · ");
  function initTrash() {
    $("tr-runs").addEventListener("change", (e) => {
      if (e.target.id === "tr-all") visibleRuns().forEach((r) => e.target.checked ? runSel.add(r.run) : runSel.delete(r.run));
      else if (e.target.dataset.rtick) e.target.checked ? runSel.add(e.target.dataset.rtick) : runSel.delete(e.target.dataset.rtick);
      renderList();
    });
    $("tr-sel-old").onclick = () => { visibleRuns().forEach((r) => r.active ? runSel.delete(r.run) : runSel.add(r.run)); renderList(); };
    $("tr-sel-none").onclick = () => { runSel.clear(); renderList(); };
    $("tr-trash").onclick = async () => {
      const list = [...runSel];
      if (!list.length) return;
      const r = await api("trash/runs", {runs: list});
      runSel.clear(); lastSig = "";
      $("tr-trash-box").open = true;
      await refresh(); loadRunTrash();
      const msg = activeMsg(r.active_changed);
      $("tr-sel-info").innerHTML = `${r.trashed.length} training${r.trashed.length > 1 ? "s" : ""} moved to the trash (${fmtBytes(r.bytes)})` + (msg ? ` · <span class="v-warn">${esc(msg)}</span>` : "");
    };
    $("tr-trash-list").addEventListener("change", (e) => {
      if (e.target.id === "trt-all") runTrash.forEach((r) => e.target.checked ? runTrashSel.add(r.run) : runTrashSel.delete(r.run));
      else if (e.target.dataset.rttick) e.target.checked ? runTrashSel.add(e.target.dataset.rttick) : runTrashSel.delete(e.target.dataset.rttick);
      document.querySelectorAll("#tr-trash-list input[data-rttick]").forEach((i) => i.checked = runTrashSel.has(i.dataset.rttick));
      runTrashInfo();
    });
    $("tr-restore").onclick = async () => {
      const r = await api("trash/restore", {runs: [...runTrashSel]});
      runTrashSel.clear(); lastSig = "";
      await refresh(); loadRunTrash();
      const msg = activeMsg(r.active_changed);
      if (msg) $("tr-sel-info").innerHTML = `<span class="v-warn">${esc(msg)}</span>`;
    };
    const purge = async (items) => {
      const n = items ? items.length : runTrash.length;
      if (!n || !window.confirm(`Delete ${n} training${n > 1 ? "s" : ""} FOR GOOD? This cannot be undone.`)) return;
      await api("trash/empty", {what: "runs", items, confirm: true});
      runTrashSel.clear(); loadRunTrash();
    };
    $("tr-purge").onclick = () => purge([...runTrashSel]);
    $("tr-empty").onclick = () => purge(undefined);
    document.querySelector('nav button[data-tab="train"]').addEventListener("click", loadRunTrash);
    loadRunTrash();
  }

  // ---- charts ----------------------------------------------------------------------------
  function barsChart(m) {
    const W = 560, H = 210, L = 150, R = 20, rows = [
      ["Entraînement", m.train_rmse_rad, "c-train"], [m.cv ? "Validation (parts)" : "Validation", m.val_rmse_rad, "c-val"], ["Geste moyen (réf.)", m.val_baseline_label_mean_rmse_rad, "c-ref"]];
    const b = m.val_baseline_label_mean_rmse_rad;
    const mx = Math.max(...rows.map((r) => r[1] || 0), b ? b * 1.15 : 0, 1e-6) * 1.08;
    const X = (v) => L + v / mx * (W - L - R);
    const o = [`<svg viewBox="0 0 ${W} ${H}" class="chart">`];
    if (b) {  // expected zone: validation well under the mean gesture
      o.push(`<rect x="${L}" y="18" width="${X(0.9 * b) - L}" height="${H - 58}" class="z-good"/>`,
        `<rect x="${X(0.9 * b)}" y="18" width="${X(b) - X(0.9 * b)}" height="${H - 58}" class="z-warn"/>`,
        `<rect x="${X(b)}" y="18" width="${W - R - X(b)}" height="${H - 58}" class="z-bad"/>`,
        `<line x1="${X(0.9 * b)}" x2="${X(0.9 * b)}" y1="14" y2="${H - 38}" class="goal"/>`,
        `<text x="${X(0.9 * b)}" y="12" class="ax" text-anchor="middle">objectif validation ≤ 0,9 × réf. = ${fr(0.9 * b)}</text>`);
    }
    rows.forEach(([lab, v, c], i) => {
      const y = 30 + i * 48;
      o.push(`<text x="${L - 8}" y="${y + 14}" class="lab" text-anchor="end">${lab}</text>`);
      if (v !== undefined) o.push(`<rect x="${L}" y="${y}" width="${Math.max(1, X(v) - L)}" height="26" rx="3" class="${c}"/>`,
        `<text x="${X(v) + 6}" y="${y + 18}" class="val">${fr(v)}</text>`);
    });
    // axis
    for (let i = 0; i <= 4; i++) { const v = mx * i / 4; o.push(`<text x="${X(v)}" y="${H - 22}" class="ax" text-anchor="middle">${fr(v, 2)}</text>`); }
    o.push(`<text x="${(L + W - R) / 2}" y="${H - 4}" class="ax" text-anchor="middle">RMSE (rad) — plus court = mieux</text></svg>`);
    return o.join("");
  }

  function curveChart(h, best) {
    const W = 560, H = 210, L = 48, R = 14, T = 14, B = 36;
    if (!h || !h.length) return `<p class="hint">Courbe en cours de chargement…</p>`;
    const vals = h.flatMap((p) => [p.train_loss, p.val_loss, p.val_loss_min, p.val_loss_max]).filter((v) => isFinite(v) && v > 0);
    let lo = Math.log10(Math.min(...vals)), hi = Math.log10(Math.max(...vals));
    if (hi - lo < 0.2) { hi += 0.1; lo -= 0.1; }
    const e1 = h[h.length - 1].epoch || 1;
    const X = (e) => L + e / e1 * (W - L - R), Y = (v) => T + (hi - Math.log10(v)) / (hi - lo) * (H - T - B);
    const path = (k) => h.filter((p) => isFinite(p[k]) && p[k] > 0).map((p, i) => `${i ? "L" : "M"}${X(p.epoch).toFixed(1)} ${Y(p[k]).toFixed(1)}`).join("");
    const o = [`<svg viewBox="0 0 ${W} ${H}" class="chart">`];
    for (let d = Math.floor(lo); d <= Math.ceil(hi); d++) for (const f of [1, 2, 5]) {
      const v = f * 10 ** d, lv = Math.log10(v);
      if (lv < lo || lv > hi) continue;
      o.push(`<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/><text x="${L - 4}" y="${Y(v) + 4}" class="ax" text-anchor="end">${String(+v.toPrecision(2)).replace(".", ",")}</text>`);
    }
    if (best !== undefined) o.push(`<line x1="${X(best)}" x2="${X(best)}" y1="${T}" y2="${H - B}" class="goal"/><text x="${X(best) + 4}" y="${T + 10}" class="ax">modèle gardé (${best})</text>`);
    // cross-validation: spread of the val loss over the parts (min..max)
    const band = h.filter((p) => isFinite(p.val_loss_min) && p.val_loss_min > 0 && isFinite(p.val_loss_max) && p.val_loss_max > 0);
    if (band.length > 1) o.push(`<path d="${band.map((p, i) => `${i ? "L" : "M"}${X(p.epoch).toFixed(1)} ${Y(p.val_loss_max).toFixed(1)}`).join("")}${band.slice().reverse().map((p) => `L${X(p.epoch).toFixed(1)} ${Y(p.val_loss_min).toFixed(1)}`).join("")}Z" class="b-val"/>`);
    o.push(`<path d="${path("train_loss")}" class="l-train"/><path d="${path("val_loss")}" class="l-val"/>`);
    const step = [10, 20, 50, 100, 200, 250, 500, 1000, 2000, 5000].find((x) => x >= e1 / 5) || 10000;
    for (let e = 0; e <= e1; e += step) o.push(`<text x="${X(e)}" y="${H - B + 16}" class="ax" text-anchor="middle">${e}</text>`);
    o.push(`<text x="${(L + W) / 2}" y="${H - 4}" class="ax" text-anchor="middle">époque · perte (échelle log)</text></svg>`);
    return o.join("");
  }

  function jointsChart(m) {
    const v = m.val_rmse_per_joint || {}, t = m.train_rmse_per_joint || {}, b = m.val_baseline_per_joint || {};
    const names = Object.keys(v).length ? Object.keys(v) : Object.keys(t);
    if (!names.length) return `<p class="hint">Pas de détail par articulation.</p>`;
    const W = 560, rowH = Math.max(16, Math.min(24, 180 / names.length)), L = 130, R = 50, H = names.length * rowH + 30;
    const mx = Math.max(...names.flatMap((n) => [v[n] || 0, t[n] || 0, b[n] || 0]), 1e-6) * 1.05;
    const X = (x) => L + x / mx * (W - L - R);
    const o = [`<svg viewBox="0 0 ${W} ${H}" class="chart">`];
    names.forEach((n, i) => {
      const y = 4 + i * rowH, bad = b[n] !== undefined && v[n] !== undefined && v[n] > 0.9 * b[n];
      o.push(`<text x="${L - 6}" y="${y + rowH * 0.62}" class="lab" text-anchor="end">${esc(n)}</text>`);
      if (v[n] !== undefined) o.push(`<rect x="${L}" y="${y + 2}" width="${Math.max(1, X(v[n]) - L)}" height="${rowH - 6}" rx="2" class="c-val${bad ? " over" : ""}"/>`,
        `<text x="${W - 4}" y="${y + rowH * 0.62}" class="val" text-anchor="end">${fr(v[n])}</text>`);
      if (t[n] !== undefined) o.push(`<circle cx="${X(t[n])}" cy="${y + rowH / 2 - 1}" r="3.5" class="c-train"/>`);
      if (b[n] !== undefined) o.push(`<line x1="${X(b[n])}" x2="${X(b[n])}" y1="${y}" y2="${y + rowH - 2}" class="tick-ref"/>`);
    });
    o.push(`<text x="${(L + W) / 2}" y="${H - 6}" class="ax" text-anchor="middle">RMSE (rad) par articulation</text></svg>`);
    return o.join("");
  }

  function foldsChart(cv) {
    // one row per part: validation bar, mean gesture tick; last row: mean ± spread
    const f = cv.per_fold, W = 560, rowH = Math.max(18, Math.min(28, 170 / (f.length + 1))), L = 110, R = 60, H = (f.length + 1) * rowH + 40;
    const mx = Math.max(...f.flatMap((r) => [r.val_rmse_rad, r.val_baseline_rmse_rad]), cv.val_rmse_mean + cv.val_rmse_std, 1e-6) * 1.08;
    const X = (x) => L + x / mx * (W - L - R);
    const o = [`<svg viewBox="0 0 ${W} ${H}" class="chart">`];
    f.forEach((r, i) => {
      const y = 6 + i * rowH, good = r.val_rmse_rad <= 0.9 * r.val_baseline_rmse_rad;
      o.push(`<text x="${L - 6}" y="${y + rowH * 0.62}" class="lab" text-anchor="end">part ${r.fold} · ${r.n_val} pr.</text>`,
        `<rect x="${L}" y="${y + 2}" width="${Math.max(1, X(r.val_rmse_rad) - L)}" height="${rowH - 6}" rx="2" class="c-val${good ? "" : " over"}"/>`,
        `<line x1="${X(r.val_baseline_rmse_rad)}" x2="${X(r.val_baseline_rmse_rad)}" y1="${y}" y2="${y + rowH - 2}" class="tick-ref"/>`,
        `<text x="${W - 4}" y="${y + rowH * 0.62}" class="val" text-anchor="end">${r.ratio ? "×" + fr(r.ratio, 2) : ""}</text>`);
    });
    const y = 10 + f.length * rowH, c = y + rowH / 2 - 2, m = cv.val_rmse_mean, s = cv.val_rmse_std;
    o.push(`<text x="${L - 6}" y="${y + rowH * 0.62}" class="lab" text-anchor="end"><tspan font-weight="600">moyenne ± écart</tspan></text>`,
      `<line x1="${X(Math.max(0, m - s))}" x2="${X(m + s)}" y1="${c}" y2="${c}" class="whisk"/>`,
      `<line x1="${X(Math.max(0, m - s))}" x2="${X(Math.max(0, m - s))}" y1="${c - 5}" y2="${c + 5}" class="whisk"/><line x1="${X(m + s)}" x2="${X(m + s)}" y1="${c - 5}" y2="${c + 5}" class="whisk"/>`,
      `<circle cx="${X(m)}" cy="${c}" r="5" class="c-mean"/>`,
      `<line x1="${X(cv.baseline_mean)}" x2="${X(cv.baseline_mean)}" y1="${y}" y2="${y + rowH - 2}" class="tick-ref"/>`,
      `<text x="${W - 4}" y="${y + rowH * 0.62}" class="val" text-anchor="end">${cv.ratio_mean ? "×" + fr(cv.ratio_mean, 2) : ""}</text>`);
    for (let i = 0; i <= 4; i++) { const v = mx * i / 4; o.push(`<text x="${X(v)}" y="${H - 18}" class="ax" text-anchor="middle">${fr(v, 2)}</text>`); }
    o.push(`<text x="${(L + W - R) / 2}" y="${H - 3}" class="ax" text-anchor="middle">RMSE de validation (rad) · ×ratio = validation ÷ geste moyen (sous 0,9 = bon)</text></svg>`);
    return o.join("");
  }

  function trendChart() {
    // val / reference ratio of every run, chronological, coloured by group
    const pts = runs.map((r, i) => ({r, i, m: r.metrics || {}})).filter((p) => p.m.val_rmse_rad !== undefined && p.m.val_baseline_label_mean_rmse_rad);
    if (pts.length < 2) return "";
    const W = 1140, H = 120, L = 44, R = 12, T = 8, B = 20;
    const ys = pts.map((p) => p.m.val_rmse_rad / p.m.val_baseline_label_mean_rmse_rad);
    const lo = Math.min(0.75, ...ys) - 0.03, hi = Math.max(1.1, ...ys) + 0.03;
    const X = (i) => L + i / Math.max(runs.length - 1, 1) * (W - L - R), Y = (v) => T + (hi - v) / (hi - lo) * (H - T - B);
    const o = [`<svg viewBox="0 0 ${W} ${H}" class="chart trend">`,
      `<rect x="${L}" y="${Y(0.9)}" width="${W - L - R}" height="${Y(lo) - Y(0.9)}" class="z-good"/>`,
      `<rect x="${L}" y="${Y(1)}" width="${W - L - R}" height="${Y(0.9) - Y(1)}" class="z-warn"/>`,
      `<rect x="${L}" y="${T}" width="${W - L - R}" height="${Y(1) - T}" class="z-bad"/>`,
      `<text x="${L - 4}" y="${Y(1) + 4}" class="ax" text-anchor="end">1,0</text><text x="${L - 4}" y="${Y(0.9) + 4}" class="ax" text-anchor="end">0,9</text>`];
    for (const g of Object.keys(GROUPS)) {
      const gp = pts.filter((p) => p.r.component === g);
      if (gp.length > 1) o.push(`<path d="${gp.map((p, k) => `${k ? "L" : "M"}${X(p.i)} ${Y(p.m.val_rmse_rad / p.m.val_baseline_label_mean_rmse_rad)}`).join("")}" style="stroke:${GCOL[g]}" class="trend-l"/>`);
    }
    pts.forEach((p) => o.push(`<circle cx="${X(p.i)}" cy="${Y(p.m.val_rmse_rad / p.m.val_baseline_label_mean_rmse_rad)}" r="${p.r.run === selected ? 7 : 4.5}" style="fill:${GCOL[p.r.component]}" class="trend-p${p.r.run === selected ? " on" : ""}" data-run="${esc(p.r.run)}"><title>${esc(p.r.run)}</title></circle>`));
    o.push(`<text x="${L}" y="${H - 4}" class="ax">validation ÷ geste moyen, du plus ancien au plus récent (sous 0,9 = bon) ·</text>`);
    let x = L + 470;
    for (const [g, l] of Object.entries(GROUPS)) { o.push(`<circle cx="${x}" cy="${H - 8}" r="4" style="fill:${GCOL[g]}"/><text x="${x + 8}" y="${H - 4}" class="ax">${l}</text>`); x += 70; }
    o.push("</svg>");
    return o.join("");
  }

  function details(m) {
    const g = m.guardian, gt = m.guardian_takes, s = m.settings;
    return `exemples ${m.n_examples} (entraînement ${m.n_train} / validation ${m.n_val}, groupes ${m.n_train_groups}/${m.n_val_groups})\n` +
      (m.reconstruction_rmse_rad !== undefined && m.reconstruction_rmse_rad !== null ? `détail perdu avant l'apprentissage (prises vs courbes à ${(s && s.n_control_points) || "?"} points) ${fr(m.reconstruction_rmse_rad)} rad : non compté dans val/entr. ci-dessous\n` : "") +
      (m.val_duration_mae_s !== undefined ? `erreur de durée (val.) ${fr(m.val_duration_mae_s)} s · (entr.) ${fr(m.train_duration_mae_s)} s\n` : "") +
      (s ? `réglages : réseau ${s.hidden.join("-")} · dropout ${s.dropout} · weight decay ${s.weight_decay} · lr ${s.lr} · lissage ${s.smoothness_weight}${s.n_control_points ? " · " + s.n_control_points + " points de contrôle" : ""} · ${s.epochs} époques${s.mirror !== undefined ? " · miroir " + (s.mirror ? "oui" : "non") : ""}\n` : "") +
      (m.mirror && m.mirror.enabled ? (m.mirror.unavailable ? `miroir gauche/droite : indisponible (${m.mirror.unavailable})\n` :
        `miroir gauche/droite : +${m.mirror.added} prises miroir à l'entraînement (${m.mirror.eligible} prises sans événement, ${m.mirror.dropped} miroirs écartés, ${m.mirror.skipped_with_event} prises avec événement non miroitées)\n` +
        (m.mirror.dropped_takes && m.mirror.dropped_takes.length ? m.mirror.dropped_takes.map((d) => `  écarté : ${d.take_id} : ${d.reason}`).join("\n") + "\n" : "")) : "") +
      (g ? `guardian (au jeu seulement, jamais utilisé par l'entraînement) : ${g.accepted}/${g.total} gestes générés jouables` +
        (g.raw_accepted !== undefined ? ` (bruts ${g.raw_accepted}/${g.total}, ralentissement médian ×${g.median_slow_down})` : "") + " " + JSON.stringify(g.violations_by_code) + "\n" : "") +
      (gt ? `guardian sur les prises : ${gt.raw_accepted}/${gt.total} telles quelles, ${gt.accepted_playable}/${gt.total} une fois rendues jouables` +
        (gt.median_excess_rad !== undefined && gt.median_excess_rad !== null ? ` (dépassement médian ${gt.median_excess_rad} rad, max ${gt.max_excess_rad} rad)` : "") +
        (gt.calibration ? ` · calibration ${gt.calibration}` : "") + "\n" : "") +
      (m.cv ? `validation croisée : ${m.cv.folds} parts (seed ${m.cv.seed}) ; validation = chaque prise prédite par le modèle qui ne l'a pas vue\n` +
        m.cv.per_fold.map((f) => `  part ${f.fold} : ${f.n_val} prises de validation, ${f.n_train} d'entraînement (+${f.n_mirrored} miroirs) · val ${fr(f.val_rmse_rad)} · réf ${fr(f.val_baseline_rmse_rad)} · entr ${fr(f.train_rmse_rad)} · meilleure époque ${f.best_epoch}`).join("\n") + "\n" +
        `modèle gardé : toutes les prises, ${m.cv.final_epochs} époques, erreur sur ses propres prises ${fr(m.final_train_rmse_rad)}\n` +
        (m.cv.warning ? "⚠ " + m.cv.warning + "\n" : "") : "") +
      (m.warning ? "⚠ " + m.warning : "");
  }

  function guardianBars(m) {
    const row = (lab, a, n) => {
      const r = n ? a / n : 0, c = r >= 0.9 ? "v-good" : r >= 0.5 ? "v-warn" : "v-bad";
      return `<div class="gbar"><span>${lab}</span><div class="gtrack"><div class="gfill ${c}" style="width:${(r * 100).toFixed(1)}%"></div><div class="gmark" style="left:90%"></div></div><b>${a}/${n}</b></div>`;
    };
    const g = m.guardian, gt = m.guardian_takes;
    return (g ? row("Gestes générés jouables", g.accepted, g.total) : "") +
      (gt && gt.total ? row("Prises jouables (après adaptation)", gt.accepted_playable, gt.total) : "") +
      (g || gt ? `<div class="legend">Repère : ≥ 90 % (trait). Le guardian protège le jeu sur le robot ; il n'influence pas l'apprentissage.</div>` : `<p class="hint">Pas de mesure guardian.</p>`);
  }

  function renderView() {
    const el = $("tr-view"), r = runs.find((x) => x.run === selected);
    if (!r) { el.innerHTML = `<p class="hint">Aucun entraînement pour l'instant.</p>`; return; }
    const m = r.metrics || {}, a = assess(m);
    el.innerHTML = `<div class="tr-head"><span class="dot" style="background:${GCOL[r.component]}"></span><b>${GROUPS[r.component] || esc(r.component)}</b> · ${fmtStamp(r.stamp)} · <span class="mono">${esc(r.run)}</span>
        <button id="tr-reuse" title="Remet les réglages de cet entraînement (réseau, dropout, weight decay, lr, lissage, miroir, parts, époques) dans l'onglet Réseau pour un nouvel entraînement"${r.metrics && r.metrics.settings ? "" : " disabled"}>Reprendre les réglages</button>
        <button id="tr-copy" title="Copie toutes les métriques (JSON) dans le presse-papiers">Copier</button></div>
      <div class="tr-trend">${trendChart()}</div>
      <div class="tr-charts">
        <figure><figcaption>Erreur sur les trajectoires</figcaption>${barsChart(m)}</figure>
        ${m.cv ? `<figure><figcaption>Validation croisée : chaque part <span class="key k-val"></span>validation <span class="key k-ref"></span>geste moyen</figcaption>${foldsChart(m.cv)}</figure>` : ""}
        <figure><figcaption>Courbe d'apprentissage <span class="key k-train"></span>entraînement <span class="key k-val"></span>validation${m.cv ? " (moyenne des parts, zone = écart min–max)" : ""}</figcaption><div id="tr-curve">${curveChart(hist[r.run], m.best_epoch)}</div></figure>
        <figure><figcaption>Validation par articulation <span class="key k-val"></span>validation <span class="key k-train dotk"></span>entraînement <span class="key k-ref"></span>geste moyen <span class="key k-over"></span>≥ 0,9 × réf.</figcaption>${jointsChart(m)}</figure>
        <figure><figcaption>Ce qu'on attend · verdict</figcaption>
          <ul class="checks">${a.map((x) => `<li class="${x.cls}">${esc(x.text)}</li>`).join("")}</ul>
          ${guardianBars(m)}
          <details><summary>Détails</summary><div class="mono">${esc(details(m))}</div></details>
        </figure>
      </div>`;
    $("tr-reuse").onclick = (e) => reuse(r, e.target);
    $("tr-copy").onclick = (e) => copyText(JSON.stringify({run: r.run, path: r.path, metrics: r.metrics}, null, 2), e.target);
    if (!hist[r.run]) {
      const want = r.run;
      api("train/history?run=" + encodeURIComponent(want), undefined, {quiet: true}).then((d) => {
        hist[want] = d.history;
        if (selected === want && $("tr-curve")) $("tr-curve").innerHTML = curveChart(d.history, m.best_epoch);
      }).catch(() => { if ($("tr-curve")) $("tr-curve").innerHTML = `<p class="hint">Courbe indisponible.</p>`; });
    }
  }

  // put the settings of a past run back in the Réseau tab (and the epochs field) for a new training
  function reuse(r, btn) {
    const s = (r.metrics || {}).settings;
    if (!s) return;
    setFields({hidden: (s.hidden || []).join(","), dropout: s.dropout, weight_decay: s.weight_decay, lr: s.lr,
               smoothness_weight: s.smoothness_weight, n_control_points: s.n_control_points || 8, folds: (r.metrics.cv && r.metrics.cv.requested) || s.folds || 1});  // runs before cross-validation: single split
    $("tr-mirror").checked = s.mirror === true;  // runs before the mirror option: no mirror
    if (s.epochs) $("tr-epochs").value = s.epochs;
    const b = document.querySelector(`[data-net-group="${r.component}"]`);
    if (b) b.click();
    changed();
    const old = btn.textContent; btn.textContent = "Réglages repris ✓";
    setTimeout(() => { btn.textContent = old; }, 1500);
  }

  function select(run) { selected = run; renderList(); renderView(); }

  // only the training in progress (and the last one if it failed): finished ones are in Résultats
  function renderJobs(jobs) {
    const running = jobs.filter((j) => j.state === "running");
    const last = jobs[jobs.length - 1];
    const show = running.length ? running : (last && last.state === "error" ? [last] : []);
    setHTML($("tr-jobs"), show.map((j) => {
      const p = j.progress.length ? j.progress[j.progress.length - 1] : null;
      const tot = p && p.stage && p.stage.startsWith("modèle final") ? null : j.epochs;
      const frac = p && tot ? Math.min(1, (p.epoch + 1) / tot) : 0;
      const secs = Math.round(Date.now() / 1000 - j.started);
      if (j.state === "error") return `<div class="tr-job err"><b>${GROUPS[j.component] || esc(j.component)}</b> · échec de l'entraînement : ${esc(j.error || "")}</div>`;
      return `<div class="tr-job"><span class="dot" style="background:${GCOL[j.component] || "#888"}"></span><b>En cours : ${GROUPS[j.component] || esc(j.component)}</b>` +
        (p ? `${p.stage ? " · " + esc(p.stage) : ""} · époque ${p.epoch}${tot ? " / " + tot : ""} · perte ${fr(p.train_loss, 4)}${p.val_loss !== null && p.val_loss !== undefined ? " (val. " + fr(p.val_loss, 4) + ")" : ""}` : " · préparation des données…") +
        ` · ${secs >= 60 ? Math.floor(secs / 60) + " min " : ""}${secs % 60} s` +
        (tot ? `<div class="gtrack"><div class="gfill v-good" style="width:${(frac * 100).toFixed(1)}%;background:var(--accent)"></div></div>` : "") + "</div>";
    }).join(""));
  }

  async function refresh() {
    const d = await api("train");
    renderJobs(d.jobs);
    const sig = d.runs.map((r) => r.run).join("|");
    if (sig === lastSig) return;
    const fresh = lastSig !== "" && d.runs.length > runs.length;
    lastSig = sig; runs = d.runs;
    for (const r of [...runSel]) if (!runs.some((x) => x.run === r)) runSel.delete(r);
    // follow the newest run (on first load and when a training finishes)
    if (!selected || fresh || !runs.some((r) => r.run === selected)) {
      const vis = runs.filter((r) => !filter || r.component === filter);
      selected = vis.length ? vis[vis.length - 1].run : (runs.length ? runs[runs.length - 1].run : null);
    }
    renderList(); renderView();
  }

  function train(c) {
    return api("train", {component: c, epochs: parseInt($("tr-epochs").value), include_pending: $("tr-pending").checked, overrides: {...overrides(), mirror: $("tr-mirror").checked}}).then(refresh);
  }

  function init() {
    // restore the last settings typed (this browser only)
    $("tr-mirror").checked = mirrorDefault();
    try { const v = JSON.parse(localStorage.getItem(STORE) || "{}"); for (const [k, id] of Object.entries(FIELDS)) if (v[k] !== undefined) $(id).value = v[k]; if (typeof v.mirror === "boolean") $("tr-mirror").checked = v.mirror; } catch (e) {}
    placeholders();
    Object.values(FIELDS).forEach((id) => $(id).addEventListener("input", changed));
    $("tr-preset-small").onclick = () => setFields({hidden: "32", dropout: 0.1, weight_decay: 0.01, smoothness_weight: 0});
    $("tr-preset-reset").onclick = () => { $("tr-mirror").checked = mirrorDefault(); setFields({}); };
    $("tr-mirror").addEventListener("change", changed);
    const layers = () => (effective(netGroup).hidden || groupModel(netGroup).hidden || [64]).slice();
    $("net-add-layer").onclick = () => { const h = layers(); if (h.length < 4) { h.push(h[h.length - 1]); $("tr-hidden").value = h.join(","); changed(); } };
    $("net-del-layer").onclick = () => { const h = layers(); if (h.length > 1) { h.pop(); $("tr-hidden").value = h.join(","); changed(); } };
    document.querySelectorAll("[data-net-group]").forEach((b) => b.onclick = () => {
      netGroup = b.dataset.netGroup;
      document.querySelectorAll("[data-net-group]").forEach((x) => x.classList.toggle("sel", x === b));
      placeholders(); drawNet(); fidelityView();
    });
    $("net-fid-go").onclick = measureFidelity;
    $("net-fid").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-k]"); if (tr) { $("tr-ctrl").value = tr.dataset.k; changed(); fidelityView(); } });
    document.querySelectorAll("button[data-train]").forEach((b) => b.onclick = () => train(b.dataset.train));
    document.querySelectorAll("[data-tr-filter]").forEach((b) => b.onclick = () => {
      filter = b.dataset.trFilter;
      document.querySelectorAll("[data-tr-filter]").forEach((x) => x.classList.toggle("sel", x === b));
      const vis = runs.filter((r) => !filter || r.component === filter);
      if (vis.length && !vis.some((r) => r.run === selected)) selected = vis[vis.length - 1].run;
      renderList(); renderView();
    });
    $("tr-order").onclick = () => {
      newestFirst = !newestFirst;
      $("tr-order").textContent = newestFirst ? "↑ plus récent en haut" : "↓ plus ancien en haut";
      renderList();
    };
    $("tr-runs").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-run]"); if (tr && !e.target.closest("td.sel")) select(tr.dataset.run); });
    initTrash();
    $("tr-view").addEventListener("click", (e) => { const p = e.target.closest("circle[data-run]"); if (p) select(p.dataset.run); });
    document.querySelector('nav button[data-tab="net"]').addEventListener("click", () =>
      api("progress", undefined, {quiet: true}).then((d) => { progress = d; drawNet(); }).catch(() => {}));
    setInterval(() => { if ($("tab-train").classList.contains("active")) refresh(); }, 2000);
    drawNet(); summary();
  }

  window.MLP = {refresh, overrides};
  const wait = setInterval(() => { if (typeof CFG !== "undefined" && CFG && CFG.net) { clearInterval(wait); init(); } }, 50);
})();
