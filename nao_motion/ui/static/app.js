"use strict";
const $ = (id) => document.getElementById(id);
let CFG = null, STATUS = null, lastPlan = null, currentTake = null;
const COLORS = ["#2f62d8","#d8572f","#1d8a4b","#9b3fc4","#c49b1f","#1f9bc4","#c41f6b","#6b7a1f","#555","#e08a00","#0aa","#a33"];

// readable guardian refusal: one line per violation, with the cycle segment where it happens
const SEG = {repos_vers_depart: "rest → start", vers_depart: "measured posture → start", depart_vers_geste: "start → gesture",
  geste: "GESTURE", geste_vers_depart: "gesture → start", depart_vers_repos: "start → rest", "départ": "first point"};
function violationsText(vs) {
  const n = (x) => (x == null ? "" : (+x).toFixed(3));
  return (vs || []).map((v) => `  • ${v.code}${v.joint ? " " + v.joint : ""}` +
    (v.segment ? ` in [${SEG[v.segment] || v.segment}]` : "") + (v.time_s != null ? ` at ${(+v.time_s).toFixed(2)} s` : "") +
    (v.value != null ? `: ${n(v.value)}` + (v.limit != null ? ` (limit ${n(v.limit)})` : "") : "") + ` — ${v.message}`).join("\n");
}

async function api(path, body, {quiet = false, timeoutMs = 0} = {}) {
  const opt = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
  // délai côté page : une requête bloquée ne fige jamais l'interface
  const ctl = timeoutMs ? new AbortController() : null;
  if (ctl) { opt.signal = ctl.signal; setTimeout(() => ctl.abort(), timeoutMs); }
  const r = await fetch("/api/" + path, opt);
  const d = await r.json();
  if (!r.ok) { const msg = d.error + (d.report ? "\n" + violationsText(d.report.violations) : ""); if (!quiet) alert(msg); throw new Error(msg); }
  return d;
}
const real = () => STATUS && STATUS.real_robot;
const PFX = {arms: "a", head: "h", hands: "m"};  // préfixe des éléments de chaque passage
const pid = (kind, name) => $(`${PFX[kind] || "a"}-${name}`);
const fmtInt = (x) => String(Math.round(Number(x) * 100) / 100).replace(".", ",");  // 0,7 · 0,25 (slider step 0.05)
const label = (id) => (CFG.emotions.find((e) => e.id === id) || {label: id}).label;

// ---- tracé SVG minimal -------------------------------------------------------
function plot(el, t, series, names, opts = {}) {
  const W = 1000, H = 260, P = 34;
  if (!t || !t.length) { el.innerHTML = ""; return; }
  let lo = Infinity, hi = -Infinity;
  series.forEach((s) => s.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
  if (hi - lo < 1e-3) { hi += 0.1; lo -= 0.1; }
  const t0 = t[0], t1 = t[t.length - 1] || 1;
  const X = (x) => P + (x - t0) / ((t1 - t0) || 1) * (W - 2 * P), Y = (y) => H - P - (y - lo) / (hi - lo) * (H - 2 * P);
  let s = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  if (opts.segment) s += `<rect x="${X(opts.segment[0])}" y="${P}" width="${X(opts.segment[1]) - X(opts.segment[0])}" height="${H - 2 * P}" fill="rgba(127,127,127,.12)"/>`;
  s += `<text x="4" y="${P}" font-size="11" fill="currentColor">${hi.toFixed(2)} rad</text><text x="4" y="${H - P}" font-size="11" fill="currentColor">${lo.toFixed(2)}</text>`;
  s += `<text x="${W - P - 40}" y="${H - 8}" font-size="11" fill="currentColor">${(t1 - t0).toFixed(2)} s</text>`;
  series.forEach((ys, k) => {
    const pts = ys.map((y, i) => `${X(t[i]).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
    s += `<polyline fill="none" stroke="${COLORS[k % COLORS.length]}" stroke-width="1.6" points="${pts}"${opts.dashed && opts.dashed.includes(k) ? ' stroke-dasharray="5 4"' : ""}/>`;
  });
  s += `<line id="${el.id}-cursor" x1="-10" x2="-10" y1="${P}" y2="${H - P}" stroke="var(--bad)"/>`;
  s += "</svg>";
  const leg = names.map((n, k) => `<span style="color:${COLORS[k % COLORS.length]}">■ ${n}</span>`).join(" ");
  el.innerHTML = s + `<div class="legend">${leg}</div>`;
  el._X = X;
}
const cols = (q) => q[0].map((_, j) => q.map((r) => r[j]));

// ---- état ----------------------------------------------------------------------
let statusPending = false, statusFails = 0;
async function refreshStatus() {
  if (statusPending) return;  // pas d'empilement de requêtes si le serveur tarde
  statusPending = true;
  try { STATUS = await api("status", undefined, {quiet: true, timeoutMs: 4000}); statusFails = 0; }
  catch (e) { if (++statusFails >= 3) $("status").textContent = "Serveur de l'interface injoignable…"; return; }
  finally { statusPending = false; }
  const b = STATUS.bridge || {};
  $("bridge-lost").classList.toggle("hidden", !(b.used && !b.connected && b.lost));
  if (b.lost) $("bridge-lost-msg").textContent = `(${b.address}) : ${b.lost}` +
    (b.lost_at ? ` — il y a ${Math.round(Date.now() / 1000 - b.lost_at)} s` : "") +
    ". Le pont arrête et maintient la posture côté robot ; une prise en cours est gardée comme incomplète.";
  const r = STATUS.recording;
  $("status").textContent = `${STATUS.adapter}${STATUS.real_robot ? ` (ROBOT RÉEL ${STATUS.robot_name || ""} ${STATUS.robot_address || "— adresse non fournie"})` : " (simulation)"} · ${STATUS.connected ? "connecté" : "déconnecté"} · contrôleur : ${STATUS.controller}` +
    (STATUS.last_error ? ` · erreur : ${STATUS.last_error}` : "") +
    (STATUS.event.present ? ` · événement az ${(STATUS.event.azimuth_rad * 180 / Math.PI).toFixed(0)}° el ${(STATUS.event.elevation_rad * 180 / Math.PI).toFixed(0)}°` +
      (STATUS.event.distance_m != null ? ` à ${(+STATUS.event.distance_m).toFixed(2)} m` : "") : " · aucun événement");
  $("armed").checked = STATUS.armed;
  const w = [];
  if (!STATUS.calibrated || !STATUS.poses_calibrated) w.push("Aucune calibration active (onglet Calibration) : limites de remplacement, exécution réelle refusée.");
  if (STATUS.sim_demonstration) w.push("Simulation : un démonstrateur SYNTHÉTIQUE remplace votre main ; les prises sont marquées comme synthétiques.");
  $("warn").textContent = w.join("\n");
  const txt = r.state === "none" ? "" : r.state === "countdown" ? `Départ dans ${Math.ceil(r.countdown_left)}…` :
    r.state === "recording" ? `● Enregistrement ${r.elapsed.toFixed(1)} s` : r.state === "prepared" ? "Prêt" :
    r.state === "done" ? `Terminé (${r.elapsed.toFixed(1)} s)${r.complete === false ? " — incomplet" + (r.error ? " : " + r.error : "") : ""}` : r.state + (r.error ? " : " + r.error : "");
  if (r.state !== "none") pid(r.kind, "rec").textContent = txt;
  if (r.state === "done") showDraft(r.kind);
  $("r-status").textContent = JSON.stringify(STATUS, null, 1);
  if (window.FOV) window.FOV.sync(STATUS.event);
  if (window.NAOFIG) window.NAOFIG.update();
}

let draftShown = null;
function showDraft(kind) {
  if (draftShown === kind) return;
  draftShown = kind;
  pid(kind, "draft").classList.remove("hidden");
  const plotEl = pid(kind, "plot");
  api("record/draft").then((d) => {
    if (kind === "hands") {  // aperçu lisible : ouverture des mains seulement (0 fermée .. 1 ouverte)
      const idx = d.joints.map((j, i) => (j.endsWith("Hand") ? i : -1)).filter((i) => i >= 0);
      d.q = d.q.map((row) => idx.map((i) => row[i])); d.joints = idx.map((i) => d.joints[i]);
    }
    plot(plotEl, d.t, cols(d.q), d.joints, {segment: d.segment});
    animate(plotEl, d.segment[0], d.segment[1]);
  });
}

async function saveDraft(kind, status) {
  const notes = pid(kind, "notes").value;
  const r = await api("record/save", {status, notes});
  draftShown = null;
  pid(kind, "draft").classList.add("hidden");
  const take = await api("takes/" + r.take_id);
  const box = pid(kind, "rec");
  box.textContent = `Prise ${r.take_id} : ${status}`;
  await loadTakes();
  return take;
}

// ---- bras ----------------------------------------------------------------------
function updateConsigne() {
  $("a-int-val").textContent = fmtInt($("a-intensity").value);
  const ev = window.FOV && window.FOV.current();
  $("a-consigne").textContent = `${label($("a-emotion").value)} — ${fmtInt($("a-intensity").value)}` + ($("a-intention").value ? ` (${$("a-intention").value})` : "") +
    (ev ? ` · ${window.FOV.describe(ev)}` : "");
  // tête et mains : même consigne (émotion, intensité, événement), passages indépendants
  for (const p of ["h", "m"]) {
    $(p + "-int-val").textContent = fmtInt($(p + "-intensity").value);
    $(p + "-consigne").textContent = `${label($(p + "-emotion").value)} — ${fmtInt($(p + "-intensity").value)}` +
      (ev ? ` · ${window.FOV.describe(ev)}` : "");
  }
}

// ---- prises ----------------------------------------------------------------------
let TAKES = [];
const takeSel = new Set();   // ticked takes (kept across refreshes)
const fmtBytes = (n) => n >= 1048576 ? `${(n / 1048576).toFixed(1).replace(".", ",")} MB` : `${Math.max(1, Math.round(n / 1024))} kB`;
const isIncomplete = (m) => ((m.annotations || {}).tags || []).includes("incomplete");
async function loadTakes() {
  const all = await api("takes" + ($("t-kind").value ? "?kind=" + $("t-kind").value : ""));
  TAKES = $("t-status").value ? all.filter((m) => m.status === $("t-status").value) : all;
  for (const id of [...takeSel]) if (!TAKES.some((m) => m.take_id === id)) takeSel.delete(id);
  const nExp = all.filter((m) => m.in_export).length;
  $("t-export-info").innerHTML = `<b>${nExp}</b> of these ${all.length} takes would be published by <code>export-dataset</code> ` +
    `(accepted, real robot, complete recording). The others never leave this PC; the trash is never exported.`;
  $("t-table").innerHTML = `<tr><th><input type="checkbox" id="t-all" title="Tick every take shown"></th><th>Prise</th><th>Type</th><th>Émotion</th><th>Int.</th><th>Durée</th><th>Événement</th><th>Statut</th><th title="Published by export-dataset">Export</th><th>Notes</th></tr>` +
    TAKES.map((m) => `<tr class="clickable${takeSel.has(m.take_id) ? " ticked" : ""}${currentTake && currentTake.meta.take_id === m.take_id ? " open" : ""}" data-id="${m.take_id}"><td class="sel"><input type="checkbox" data-tick="${m.take_id}"${takeSel.has(m.take_id) ? " checked" : ""}></td><td>${m.take_id}${m.synthetic ? " <span class='tag'>synth</span>" : ""}${isIncomplete(m) ? " <span class='tag'>incomplete</span>" : ""}</td><td>${m.kind}</td><td>${label(m.emotion)}</td>` +
      `<td>${fmtInt(m.intensity)}</td><td>${m.duration_s.toFixed(1)} s</td><td>${m.event.present ? `az ${(m.event.azimuth_rad * 180 / Math.PI).toFixed(0)}° el ${(m.event.elevation_rad * 180 / Math.PI).toFixed(0)}°` + (m.event.distance_m != null ? ` ${(+m.event.distance_m).toFixed(2)} m` : "") : "aucun"}</td>` +
      `<td class="${m.status}">${m.status}</td><td class="${m.in_export ? "accepted" : "muted"}">${m.in_export ? "✓" : "—"}</td><td>${(m.annotations.notes || "")}</td></tr>`).join("");
  document.querySelectorAll("#t-table tr.clickable").forEach((tr) => tr.onclick = (e) => { if (!e.target.closest("td.sel")) openTake(tr.dataset.id); });
  takeSelInfo();
  loadTakeTrash();
  renderProgress(await api("progress", undefined, {quiet: true}).catch((e) => ({error: e.message})));
  const st = await api("stats");
  let h = "";
  for (const kind of ["arms", "head", "hands"]) {
    if (!st.table[kind]) continue;
    h += `<h4>${{arms: "Bras", head: "Tête", hands: "Mains"}[kind]}</h4><table><tr><th></th>${st.bins.map((b) => `<th>${b}</th>`).join("")}</tr>`;
    for (const e of CFG.emotions) {
      h += `<tr><td>${e.label}</td>` + st.bins.map((b) => { const c = st.table[kind][e.id][b] || {}; const tot = Object.values(c).reduce((a, x) => a + x, 0); return `<td>${c.accepted || 0} / ${tot}</td>`; }).join("") + "</tr>";
    }
    h += "</table>";
  }
  $("stats").innerHTML = h;
}

// selection and trash of takes ------------------------------------------------------
function takeSelInfo() {
  const n = takeSel.size;
  $("t-trash").disabled = !n;
  $("t-trash").textContent = n ? `Move ${n} take${n > 1 ? "s" : ""} to trash` : "Move to trash";
  $("t-sel-info").textContent = n ? `${n} ticked` : "";
  const all = $("t-all");
  if (all) { all.checked = n > 0 && TAKES.every((m) => takeSel.has(m.take_id)); all.indeterminate = n > 0 && !all.checked; }
  document.querySelectorAll("#t-table tr[data-id]").forEach((tr) => {
    const on = takeSel.has(tr.dataset.id);
    tr.classList.toggle("ticked", on);
    tr.querySelector("input[data-tick]").checked = on;
  });
}
let takeTrash = [];
const trashSel = new Set();
async function loadTakeTrash() {
  const d = await api("trash", undefined, {quiet: true}).catch(() => null);
  if (!d) return;
  takeTrash = d.takes;
  for (const id of [...trashSel]) if (!takeTrash.some((t) => t.take_id === id)) trashSel.delete(id);
  $("t-trash-sum").innerHTML = `Trash: <b>${takeTrash.length}</b> take${takeTrash.length > 1 ? "s" : ""}` +
    (takeTrash.length ? ` (${fmtBytes(d.takes_bytes)}, in <code>data/trash/takes/</code>, not exported, restorable)` : " (empty)");
  $("t-empty").disabled = !takeTrash.length;
  $("t-trash-table").innerHTML = takeTrash.length ? `<tr><th><input type="checkbox" id="tt-all"></th><th>Prise</th><th>Type</th><th>Émotion</th><th>Int.</th><th>Statut</th><th>Size</th><th>Trashed</th></tr>` +
    takeTrash.map((t) => `<tr data-tid="${t.take_id}"><td class="sel"><input type="checkbox" data-ttick="${t.take_id}"${trashSel.has(t.take_id) ? " checked" : ""}></td><td>${t.take_id}${t.synthetic ? " <span class='tag'>synth</span>" : ""}${(t.tags || []).includes("incomplete") ? " <span class='tag'>incomplete</span>" : ""}</td><td>${t.kind || "?"}</td>` +
      `<td>${t.emotion ? label(t.emotion) : "?"}</td><td>${t.intensity != null ? fmtInt(t.intensity) : ""}</td><td class="${t.status}">${t.status || ""}</td><td>${fmtBytes(t.bytes)}</td><td>${(t.trashed_at || "").replace("T", " ")}</td></tr>`).join("") : "";
  trashSelInfo();
}
function trashSelInfo() {
  const n = trashSel.size;
  $("t-restore").disabled = $("t-purge").disabled = !n;
  $("t-restore").textContent = n ? `Restore ${n}` : "Restore selected";
  $("t-purge").textContent = n ? `Delete ${n} for good` : "Delete selected for good";
  const all = $("tt-all");
  if (all) { all.checked = n > 0 && n === takeTrash.length; all.indeterminate = n > 0 && !all.checked; }
}
function initTakeTrash() {
  $("t-status").onchange = loadTakes;
  $("t-table").addEventListener("change", (e) => {
    if (e.target.id === "t-all") { TAKES.forEach((m) => e.target.checked ? takeSel.add(m.take_id) : takeSel.delete(m.take_id)); }
    else if (e.target.dataset.tick) { e.target.checked ? takeSel.add(e.target.dataset.tick) : takeSel.delete(e.target.dataset.tick); }
    takeSelInfo();
  });
  const pick = {rejected: (m) => m.status === "rejected", pending: (m) => m.status === "pending", synthetic: (m) => m.synthetic,
    incomplete: isIncomplete, noexport: (m) => !m.in_export};
  document.querySelectorAll("[data-tsel]").forEach((b) => b.onclick = () => {
    if (b.dataset.tsel === "none") takeSel.clear();
    else TAKES.filter(pick[b.dataset.tsel]).forEach((m) => takeSel.add(m.take_id));
    takeSelInfo();
  });
  $("t-trash").onclick = async () => {
    const ids = [...takeSel];
    if (!ids.length) return;
    const r = await api("trash/takes", {ids});
    takeSel.clear();
    if (currentTake && ids.includes(currentTake.meta.take_id)) clearTakeView();
    $("t-trash-box").open = true;
    await loadTakes();
    $("t-sel-info").textContent = `${r.trashed.length} take${r.trashed.length > 1 ? "s" : ""} moved to the trash (${fmtBytes(r.bytes)})`;
  };
  $("t-trash-table").addEventListener("change", (e) => {
    if (e.target.id === "tt-all") { takeTrash.forEach((t) => e.target.checked ? trashSel.add(t.take_id) : trashSel.delete(t.take_id)); }
    else if (e.target.dataset.ttick) { e.target.checked ? trashSel.add(e.target.dataset.ttick) : trashSel.delete(e.target.dataset.ttick); }
    document.querySelectorAll("#t-trash-table input[data-ttick]").forEach((i) => i.checked = trashSel.has(i.dataset.ttick));
    trashSelInfo();
  });
  $("t-restore").onclick = async () => { await api("trash/restore", {takes: [...trashSel]}); trashSel.clear(); loadTakes(); };
  const purge = async (items) => {
    const n = items ? items.length : takeTrash.length;
    if (!n || !window.confirm(`Delete ${n} take${n > 1 ? "s" : ""} FOR GOOD? This cannot be undone.`)) return;
    await api("trash/empty", {what: "takes", items, confirm: true});
    trashSel.clear(); loadTakeTrash();
  };
  $("t-purge").onclick = () => purge([...trashSel]);
  $("t-empty").onclick = () => purge(undefined);
}

// barre de progression à paliers par groupe (prises acceptées) -------------------
function renderProgress(pr) {
  const box = $("progress");
  if (!pr || !pr.kinds) { box.innerHTML = pr && pr.error ? `<div class="alert"><b>Avancement indisponible :</b> ${pr.error}</div>` : ""; return; }
  const pct = (n, goal) => Math.min(100, (100 * n) / goal);
  box.innerHTML = ["arms", "head", "hands"].map((kind) => {
    const k = pr.kinds[kind];
    if (!k) return "";
    const goal = k.goal, acc = k.accepted;
    const ticks = k.milestones.map((m) =>
      `<span class="pg-tick${m.reached ? " reached" : ""}" style="left:${pct(m.takes, goal)}%" title="${m.takes} : ${m.label}"><b>${m.takes}</b></span>`).join("");
    const pend = Math.min(k.pending, Math.max(0, goal - acc));
    const msg = k.next
      ? `Encore <b>${k.next.remaining}</b> prise${k.next.remaining > 1 ? "s" : ""} acceptée${k.next.remaining > 1 ? "s" : ""} avant le palier <b>${k.next.takes}</b> (${k.next.label})`
      : `<b>Objectif atteint</b> (${goal} prises)`;
    const done = k.reached ? ` · palier atteint : ${k.reached.takes} (${k.reached.label})` : "";
    const cov = k.coverage, emos = Object.entries(cov.emotions);
    const low = emos.length ? emos.reduce((a, b) => (b[1] < a[1] ? b : a)) : null;
    const hi = emos.length ? Math.max(...emos.map((e) => e[1])) : 0;
    const lag = low && hi - low[1] >= 2 ? ` · en retard : ${pr.emotion_labels[low[0]] || low[0]} (${low[1]} contre ${hi})` : "";
    const extra = [k.pending ? `${k.pending} en attente à trier` : "", k.rejected ? `${k.rejected} rejetée${k.rejected > 1 ? "s" : ""}` : "",
      k.synthetic_accepted ? `${k.synthetic_accepted} simulée${k.synthetic_accepted > 1 ? "s" : ""} non comptée${k.synthetic_accepted > 1 ? "s" : ""}` : ""].filter(Boolean).join(" · ");
    return `<div class="pg" data-kind="${kind}">
      <div class="pg-head"><span class="pg-name">${k.label}</span><span class="pg-count"><b>${acc}</b> / ${goal}</span><span class="pg-msg">${msg}${done}</span></div>
      <div class="pg-bar" role="progressbar" aria-label="${k.label} : prises acceptées" aria-valuemin="0" aria-valuemax="${goal}" aria-valuenow="${acc}">
        <div class="pg-fill" style="width:${pct(acc, goal)}%"></div><div class="pg-pend" style="left:${pct(acc, goal)}%;width:${pct(pend, goal)}%" title="en attente"></div>${ticks}
      </div>
      <div class="pg-sub">Cases couvertes : <b>${cov.event_covered}</b> / ${cov.event_cells} avec événement · ${cov.none_covered} / ${cov.none_cells} sans événement${lag}${extra ? " · " + extra : ""}</div>
    </div>`;
  }).join("");
}

// openTake (viewer of the selected take, right of the list): takeview.js

function animate(el, t0, t1) {
  const line = document.getElementById(el.id + "-cursor");
  if (!line || !el._X) return;
  const start = performance.now();
  const step = (now) => {
    const t = t0 + (now - start) / 1000;
    const x = el._X(Math.min(t, t1));
    line.setAttribute("x1", x); line.setAttribute("x2", x);
    if (t < t1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ---- entraînement ---------------------------------------------------------------
// réécrit le bloc seulement si son contenu change : une sélection en cours (copier-coller) n'est pas perdue
function setHTML(el, html) {
  if (el._html === html) return;
  el._html = html;
  el.innerHTML = html;
}

async function copyText(text, btn) {
  try { await navigator.clipboard.writeText(text); }
  catch (e) {  // repli sans API presse-papiers
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    document.execCommand("copy"); ta.remove();
  }
  const old = btn.textContent; btn.textContent = "Copié ✓";
  setTimeout(() => { btn.textContent = old; }, 1500);
}
// training tab and network tab: mlp.js (window.MLP)
const refreshTrain = () => window.MLP && window.MLP.refresh();

// ---- génération -----------------------------------------------------------------
// consigne of the Generation tab (also used by the latent space view, latent.js)
function genConsigne() {
  const e1 = $("g-emotion").value, e2 = $("g-emotion2").value, w = parseFloat($("g-mix").value);
  const body = {emotion: e1, intensity: parseFloat($("g-intensity").value), use_event: $("g-event").checked};
  if (e2 && e2 !== e1) body.mix = {[e1]: 1 - w, [e2]: w};
  return body;
}
// extra: CVAE fields from the latent space view ({latent, emotion, intensity, event, mix: null...})
async function generate(extra) {
  const body = {...genConsigne(), variation: parseFloat($("g-var").value), project_zone: $("g-project").checked};
  const temp = $("g-temp").value.trim();
  if (temp !== "") body.temperature = parseFloat(temp);
  if (extra && typeof extra === "object" && !(extra instanceof Event)) Object.assign(body, extra);
  if (body.mix === null) delete body.mix;
  const out = await api("generate", body);
  const tr = out.played || out.generated.trajectory;  // the gesture that would be played
  plot($("g-plot"), tr.t, cols(tr.q), tr.joints);
  animate($("g-plot"), 0, tr.t[tr.t.length - 1]);
  const mr = out.motion_report, cy = out.cycle;
  const ad = out.adaptation;
  $("g-report").textContent = `geste : ${(tr.t[tr.t.length - 1] - tr.t[0]).toFixed(2)} s · guardian : ${mr.ok ? "accepté" : "REFUSÉ"}\n` +
    (ad && ad.changed ? `made playable: ${ad.notes.join(" ; ")} (raw gesture ${out.generated.duration_s.toFixed(2)} s, guardian on raw: ${out.raw_report.ok ? "accepted" : "refused"})\n` : "") +
    (mr.violations.length ? violationsText(mr.violations) + "\n" : "") +
    (cy ? `cycle complet (${cy.start === "rest" ? "from rest" : "from the current posture"}, if played now) : ${cy.report.ok ? "accepté" : "REFUSÉ"} (${cy.duration_s.toFixed(2)} s) ${cy.report.violations.length ? "\n" + violationsText(cy.report.violations) : ""}` : (out.cycle_error ? "cycle : " + out.cycle_error : "")) +
    (out.generated.inputs.latent ? `\nCVAE : ${body.latent ? "z choisi sur la carte (" + Object.keys(body.latent).join(", ") + "), autres groupes : " : ""}température ${out.generated.inputs.temperature} · ` + Object.entries(out.generated.inputs.latent).map(([g, z]) => `${g} z = [${z.map((v) => v.toFixed(2)).join(", ")}]`).join(" · ") : "") +
    (out.generated.notes.length ? "\nnotes : " + out.generated.notes.join(" ; ") : "");
  if (window.LATENT) window.LATENT.onGenerated(out);
  lastPlan = out.plan_id; lastGen = out;
  updatePlay();
  return out;
}

// ---- Generation tab: play the generated gesture on the robot ----------------------
let lastGen = null, playing = false;
// why Play is unavailable right now ("" when it can be pressed)
function playBlocker(noGesture = false) {
  if (!lastGen && !noGesture) return "Generate a gesture first (key 1).";
  if (!STATUS) return "Waiting for the robot status…";
  if (!STATUS.connected) return "Robot not connected (Robot tab).";
  if (!STATUS.armed) return "Motors not armed: tick « moteurs armés » at the top of the page.";
  if (STATUS.controller === "executing" || playing) return "A motion is already being played.";
  if (STATUS.controller === "fault") return `Controller in FAULT: ${STATUS.last_error || "see the log (Robot tab)"}\n` +
    "Check the robot, then « Acknowledge » (no motion) and re-arm.";
  if (STATUS.recording && ["countdown", "recording"].includes(STATUS.recording.state)) return "Recording in progress.";
  return "";
}
const RELAXED_END = "\nMotors relaxed at the end.";
function updatePlay() {
  const why = playBlocker(), btn = $("g-play");
  if (!btn) return;
  btn.disabled = !!why;
  $("g-rest-relax").disabled = !!playBlocker(true);
  $("g-hold").disabled = !(STATUS && STATUS.connected) || STATUS.controller === "executing";
  const p = STATUS && STATUS.play;
  let txt = why;
  if (p && p.kind === "rest") {
    txt = (p.state === "playing" ? `↓ Going back to rest (${(+p.duration_s).toFixed(1)} s), then relaxing…` :
      p.state === "done" ? "✓ At rest, motors relaxed." :
      p.state === "relaxed" ? "Relaxed during the return to rest (the motion ended there)." :
      p.state === "stopped" ? "■ Stopped: posture held." :
      `✗ ${p.error}`) + (playBlocker(true) && p.state !== "playing" ? "\n" + playBlocker(true) : "");
  } else if (p && lastGen && p.plan_id === lastGen.plan_id) {
    const from = p.start === "rest" ? "from rest" : "from the current posture";
    txt = (p.state === "playing" ? `▶ Playing ${from} (${(+p.duration_s).toFixed(1)} s)…` :
      p.state === "done" ? `✓ Played ${from} (${(+p.duration_s).toFixed(1)} s).` + (p.relaxed_end ? RELAXED_END : "") :
      p.state === "relaxed" ? "Relaxed during the play (the motion ended there)." :
      p.state === "stopped" ? "■ Stopped: posture held." :
      p.state === "refused" ? `✗ ${p.error}` : `✗ Error: ${p.error}`) + (why && p.state !== "playing" ? "\n" + why : "");
    if (p.warnings && p.warnings.length) txt += "\n⚠ tracking (tolerated): " + p.warnings.join(" ; ");
    if (p.guardian === false) txt += "\nGUARDIAN OFF: raw gesture" + (p.guardian_would ? ` (the guardian would have ${p.guardian_would.ok ? "accepted" : "REFUSED"} it)` : "") +
      (p.notes && p.notes.length ? " · " + p.notes.join(" ; ") : "");
  }
  $("g-play-why").textContent = txt;
  $("g-ack").classList.toggle("hidden", !(STATUS && STATUS.controller === "fault"));
}
async function play() {
  if (playBlocker()) { updatePlay(); return; }
  playing = true; $("g-play").disabled = true;
  try {
    const r = await api("play", {plan_id: lastPlan, start: $("g-start").value, end: $("g-end").value,
      guardian: !$("g-noguard").checked, relax_end: $("g-relax-end").checked}, {quiet: true});
    STATUS.play = {state: "playing", plan_id: lastPlan, start: r.start, end: r.end, duration_s: r.duration_s,
      guardian: r.guardian, notes: r.notes, guardian_would: r.guardian ? null : {ok: r.cycle.report.ok}};
    const c = r.cycle, tr = c && lastGen && (lastGen.played || lastGen.generated.trajectory);
    if (tr) {  // cursor over the gesture while the whole cycle plays
      const g = c.segments.find((x) => x[0] === "geste");
      if (g) setTimeout(() => animate($("g-plot"), 0, tr.t[tr.t.length - 1]), g[1] * 1000);
    }
  } catch (e) {
    $("g-play-why").textContent = "✗ " + e.message;
    playing = false; return;
  }
  playing = false; updatePlay();
}

// Generation tab, key 3: motorised return to rest from the measured posture, then every motor relaxed
async function restRelax() {
  if (playBlocker(true)) { updatePlay(); return; }
  playing = true; $("g-rest-relax").disabled = true;
  try {
    const r = await api("rest_relax", {}, {quiet: true});
    STATUS.play = {state: "playing", kind: "rest", duration_s: r.duration_s};
  } catch (e) {
    $("g-play-why").textContent = "✗ " + e.message;
  }
  playing = false; updatePlay();
}

// ---- liaisons -------------------------------------------------------------------
async function init() {
  CFG = await api("config");
  const opts = CFG.emotions.map((e) => `<option value="${e.id}">${e.label}</option>`).join("");
  $("a-emotion").innerHTML = opts; $("g-emotion").innerHTML = opts; $("h-emotion").innerHTML = opts; $("m-emotion").innerHTML = opts;
  ["h-emotion", "h-intensity", "m-emotion", "m-intensity"].forEach((id) => $(id).oninput = updateConsigne);
  $("g-emotion2").innerHTML += opts;
  $("a-intention").innerHTML += CFG.intentions.map((i) => `<option>${i}</option>`).join("");
  ["a-emotion", "a-intensity", "a-intention"].forEach((id) => $(id).oninput = updateConsigne);
  $("g-intensity").oninput = () => $("g-int-val").textContent = fmtInt($("g-intensity").value);
  updateConsigne(); $("g-int-val").textContent = fmtInt(0.7);
  document.querySelectorAll("nav button").forEach((b) => b.onclick = () => {
    document.querySelectorAll("nav button, .tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active"); $("tab-" + b.dataset.tab).classList.add("active");
    if (b.dataset.tab === "train") refreshTrain();
    if (["takes", "head", "hands"].includes(b.dataset.tab)) loadTakes();
    if (window.NAOFIG) window.NAOFIG.update();
  });
  $("stop").onclick = () => api("stop", {});
  // sans confirmation (préférence de l'opérateur) : avertissement permanent en haut de page
  $("armed").onchange = async (e) => {
    const on = e.target.checked;
    if (on && !window.confirm("Autoriser les mouvements motorisés ? Chaque mouvement restera déclenché par un bouton explicite.")) { e.target.checked = false; return; }
    await api("arm", {armed: on, confirm: on}).catch(() => (e.target.checked = false));
  };
  // bras
  // pas de confirmation (souhait de Monolithoid) : l'avertissement de la page (#a-alert) la remplace
  $("a-prepare").onclick = () =>
    api("record/prepare", {kind: "arms", emotion: $("a-emotion").value, intensity: parseFloat($("a-intensity").value), intention: $("a-intention").value, confirm: true}).then(() => { draftShown = null; });
  $("a-start").onclick = () => api("record/start", {});
  $("a-stop").onclick = () => api("record/stop", {});
  $("a-hold").onclick = () => api("hold", {groups: ["left_arm", "right_arm"]});
  $("a-accept").onclick = () => saveDraft("arms", "accepted");
  $("a-pending").onclick = () => saveDraft("arms", "pending");
  $("a-reject").onclick = () => saveDraft("arms", "rejected");
  $("a-retry").onclick = async () => { await saveDraft("arms", "rejected"); $("a-prepare").onclick(); };
  // tête (indépendante des bras) ; pas de confirmation : l'avertissement de la page (#h-alert) la remplace
  $("h-prepare").onclick = () =>
    api("record/prepare", {kind: "head", emotion: $("h-emotion").value, intensity: parseFloat($("h-intensity").value), confirm: true})
      .then(() => { draftShown = null; });
  $("h-start").onclick = () => api("record/start", {});
  $("h-stop").onclick = () => api("record/stop", {});
  $("h-hold").onclick = () => api("hold", {groups: ["head"]});
  $("h-accept").onclick = () => saveDraft("head", "accepted");
  $("h-pending").onclick = () => saveDraft("head", "pending");
  $("h-reject").onclick = () => saveDraft("head", "rejected");
  $("h-retry").onclick = async () => { await saveDraft("head", "rejected"); $("h-prepare").onclick(); };
  // mains (indépendantes des bras) : curseurs et clavier dans hands.js
  // pas de confirmation : l'avertissement de la page (#m-alert) la remplace
  $("m-prepare").onclick = () =>
    api("record/prepare", {kind: "hands", emotion: $("m-emotion").value, intensity: parseFloat($("m-intensity").value), confirm: true})
      .then(() => { draftShown = null; });
  $("m-start").onclick = () => api("record/start", {});
  $("m-stop").onclick = () => api("record/stop", {});
  $("m-hold").onclick = () => api("hold", {});
  $("m-accept").onclick = () => saveDraft("hands", "accepted");
  $("m-pending").onclick = () => saveDraft("hands", "pending");
  $("m-reject").onclick = () => saveDraft("hands", "rejected");
  $("m-retry").onclick = async () => { await saveDraft("hands", "rejected"); $("m-prepare").onclick(); };
  // prises
  $("t-kind").onchange = loadTakes; $("t-refresh").onclick = loadTakes; initTakeTrash();
  initTakeView();  // takeview.js
  // entraînement et réseau : mlp.js
  // génération
  $("g-go").onclick = () => generate();
  $("g-temp").placeholder = (CFG.cvae || {}).temperature ?? 1;
  // no confirmation (operator preference): the permanent warning #g-alert replaces it
  $("g-play").onclick = play;
  $("g-rest-relax").onclick = restRelax;
  $("g-hold").onclick = () => api("hold", {}).then(refreshStatus);
  // relaxing is always possible, also mid-gesture (the motion ends there): same as Space
  $("g-relax").onclick = () => api("relax", {groups: ["head", "left_arm", "right_arm", "hands"], confirm: true}).then(refreshStatus);
  $("g-diag-go").onclick = async () => {
    $("g-diag-out").textContent = "Running…";
    const d = await api("guardian/diagnose", {project_zone: $("g-project").checked}).catch((e) => ({error: e.message}));
    if (d.error) { $("g-diag-out").textContent = "✗ " + d.error; return; }
    const tbl = (rows) => rows.length ? rows.map((r) => `  ${String(r.gestures).padStart(3)} × ${r.code} ${r.joint}` +
      (r.segment ? ` [${SEG[r.segment] || r.segment}]` : "")).join("\n") : "  none";
    const L = d.limits;
    $("g-diag-out").textContent =
      `${d.n} gestures · accepted: raw ${d.ok.raw}/${d.n} · made playable ${d.ok.playable}/${d.n}` +
      (d.connected ? ` · whole cycle ${d.ok.cycle}/${d.n}` : " · cycle not checked (no robot connected)") +
      `\nslow-down to respect the limits: median ×${d.slow_down.median} · max ×${d.slow_down.max}` +
      (d.slow_down.capped_by_max_duration ? ` · ${d.slow_down.capped_by_max_duration} capped by the max duration` : "") +
      (() => {  // recorded takes, no model: every refusal here is a guardian error
        const tk = d.takes || {};
        if (tk.error) return `\nYOUR TAKES through the guardian: ✗ ${tk.error}`;
        if (!tk.n) return "\nYOUR TAKES through the guardian: no recorded arm take";
        const t2 = (rows) => rows.length ? rows.map((r) => `  ${String(r.takes).padStart(3)} × ${r.code} ${r.joint}` +
          (r.segment ? ` [${SEG[r.segment] || r.segment}]` : "")).join("\n") : "  none";
        return `\nYOUR TAKES through the guardian (no model): raw ${tk.ok.raw}/${tk.n} · made playable ${tk.ok.playable}/${tk.n}` +
          (tk.connected ? ` · whole cycle ${tk.ok.cycle}/${tk.n}` : "") +
          "\n  still refused after making them playable:\n" + t2(tk.refusals.playable) +
          (tk.connected ? "\n  whole cycle:\n" + t2(tk.refusals.cycle) : "") +
          (tk.refused_takes.length ? `\n  e.g. ${tk.refused_takes.slice(0, 5).join(", ")}` : "");
      })() +
      "\n\nWHAT THIS MEANS\n" + d.hints.map((x) => "• " + x).join("\n") +
      "\n\nRefused on the RAW gesture (gestures per check):\n" + tbl(d.refusals.raw) +
      "\nStill refused after making it playable:\n" + tbl(d.refusals.playable) +
      (d.connected ? "\nWhole cycle:\n" + tbl(d.refusals.cycle) : "") +
      `\n\nLimits in force (calibration ${L.calibration}${L.calibrated ? "" : ", NOT calibrated"}):\n` +
      L.joints.map((r) => `  ${r.joint.padEnd(15)} zone [${r.zone[0]}, ${r.zone[1]}] · v ${r.velocity} (${r.velocity_source}) · a ${r.acceleration} (${r.acceleration_source})`).join("\n") +
      "\nCoupled constraints:\n" + (L.coupled.map((c) => `  ${c.name} (${c.source}) : ` +
        Object.entries(c.terms).map(([j, k]) => `${k}·${j}`).join(" + ") + ` ≤ ${c.bound}`).join("\n") || "  none");
  };
  $("g-ack").onclick = () => api("acknowledge", {}).then(refreshStatus);
  $("g-project").checked = CFG.project_control_points !== false;  // default from robot.yaml
  // guardian off for a test: never remembered, red frame + permanent warning while ticked
  $("g-noguard").checked = false;
  $("g-noguard").onchange = (e) => {
    const off = e.target.checked;
    $("g-noguard-alert").classList.toggle("hidden", !off);
    document.querySelector("fieldset.g-play").classList.toggle("noguard", off);
    $("g-play").textContent = off ? "2 ▶ Play RAW, guardian off" : "2 ▶ Play on the robot";
  };
  setInterval(updatePlay, 300);
  // robot
  $("r-connect").onclick = () => api("connect", {});
  const stored = (() => { try { return JSON.parse(localStorage.getItem("nao-robot") || "{}"); } catch (e) { return {}; } })();
  $("r-host").value = stored.host || (CFG.robot && CFG.robot.ip) || "";
  $("r-port").value = stored.port || (CFG.robot && CFG.robot.port) || 9559;
  $("r-bhost").value = stored.bhost || (CFG.bridge && CFG.bridge.host) || "127.0.0.1";
  $("r-bport").value = stored.bport || (CFG.bridge && CFG.bridge.port) || 9600;
  $("r-bloc").value = stored.bloc || (CFG.bridge && CFG.bridge.location) || "robot";
  const syncBloc = () => { $("r-bhost-l").classList.toggle("hidden", $("r-bloc").value === "robot"); $("r-bport-l").classList.toggle("hidden", $("r-bloc").value !== "robot"); };
  $("r-bloc").onchange = syncBloc; syncBloc();
  // bandeau « Pont perdu » : reconnexion au même pont, sans boîte de dialogue ni mouvement
  $("bridge-reconnect").onclick = async () => {
    const msg = $("bridge-reconnect-msg");
    let v = {};
    try { v = JSON.parse(localStorage.getItem("nao-robot") || "{}"); } catch (e) {}
    v = {host: v.host || $("r-host").value.trim(), port: v.port || parseInt($("r-port").value), bhost: v.bhost || $("r-bhost").value.trim(),
         bport: v.bport || parseInt($("r-bport").value), bloc: v.bloc || $("r-bloc").value};
    msg.textContent = "reconnexion…";
    try {
      await api("connect", {mode: "robot", robot_host: v.host, robot_port: v.port, bridge_host: v.bhost, bridge_port: v.bport, bridge_on_robot: v.bloc === "robot"},
                {quiet: true, timeoutMs: 20000});
      msg.textContent = "";
    } catch (e) { msg.textContent = "échec : " + e.message; }
  };
  $("r-connect-robot").onclick = async () => {
    const v = {host: $("r-host").value.trim(), port: parseInt($("r-port").value), bhost: $("r-bhost").value.trim(), bport: parseInt($("r-bport").value), bloc: $("r-bloc").value};
    if (!v.host) return alert("Indiquer l'adresse du robot (IP ou nom DNS)");
    await api("connect", {mode: "robot", robot_host: v.host, robot_port: v.port, bridge_host: v.bhost, bridge_port: v.bport, bridge_on_robot: v.bloc === "robot"});
    try { localStorage.setItem("nao-robot", JSON.stringify(v)); } catch (e) {}
  };
  $("r-sim").onclick = () => api("connect", {mode: "sim"});
  $("r-ack").onclick = () => api("acknowledge", {});
  $("r-hold-all").onclick = () => api("hold", {});
  $("r-rest").onclick = () => window.confirm("Retour motorisé au repos depuis la posture mesurée ?") && api("rest", {confirm: true});
  $("r-relax-arms").onclick = () => window.confirm("Relâcher les bras ? Ils tomberont s'ils ne sont pas tenus.") && api("relax", {groups: ["left_arm", "right_arm"], confirm: true});
  $("r-relax-head").onclick = () => window.confirm("Relâcher la tête ?") && api("relax", {groups: ["head"], confirm: true});
  await loadTakes();
  setInterval(refreshStatus, 300);
}
init();
