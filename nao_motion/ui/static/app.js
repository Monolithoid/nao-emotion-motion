"use strict";
const $ = (id) => document.getElementById(id);
let CFG = null, STATUS = null, lastPlan = null, currentTake = null;
const COLORS = ["#2f62d8","#d8572f","#1d8a4b","#9b3fc4","#c49b1f","#1f9bc4","#c41f6b","#6b7a1f","#555","#e08a00","#0aa","#a33"];

async function api(path, body, {quiet = false, timeoutMs = 0} = {}) {
  const opt = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
  // délai côté page : une requête bloquée ne fige jamais l'interface
  const ctl = timeoutMs ? new AbortController() : null;
  if (ctl) { opt.signal = ctl.signal; setTimeout(() => ctl.abort(), timeoutMs); }
  const r = await fetch("/api/" + path, opt);
  const d = await r.json();
  if (!r.ok) { const msg = d.error + (d.report ? "\n" + JSON.stringify(d.report.violations, null, 1) : ""); if (!quiet) alert(msg); throw new Error(msg); }
  return d;
}
const real = () => STATUS && STATUS.real_robot;
const PFX = {arms: "a", head: "h", hands: "m"};  // préfixe des éléments de chaque passage
const pid = (kind, name) => $(`${PFX[kind] || "a"}-${name}`);
const fmtInt = (x) => Number(x).toFixed(1).replace(".", ",");
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
  $("warn").textContent = w.join(" ");
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
async function loadTakes() {
  TAKES = await api("takes" + ($("t-kind").value ? "?kind=" + $("t-kind").value : ""));
  $("t-table").innerHTML = "<tr><th>Prise</th><th>Type</th><th>Émotion</th><th>Int.</th><th>Durée</th><th>Événement</th><th>Statut</th><th>Notes</th></tr>" +
    TAKES.map((m) => `<tr class="clickable" data-id="${m.take_id}"><td>${m.take_id}${m.synthetic ? " <span class='tag'>synth</span>" : ""}</td><td>${m.kind}</td><td>${label(m.emotion)}</td>` +
      `<td>${fmtInt(m.intensity)}</td><td>${m.duration_s.toFixed(1)} s</td><td>${m.event.present ? `az ${(m.event.azimuth_rad * 180 / Math.PI).toFixed(0)}° el ${(m.event.elevation_rad * 180 / Math.PI).toFixed(0)}°` + (m.event.distance_m != null ? ` ${(+m.event.distance_m).toFixed(2)} m` : "") : "aucun"}</td>` +
      `<td class="${m.status}">${m.status}</td><td>${(m.annotations.notes || "")}</td></tr>`).join("");
  document.querySelectorAll("#t-table tr.clickable").forEach((tr) => tr.onclick = () => openTake(tr.dataset.id));
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

async function openTake(id) {
  currentTake = await api("takes/" + id);
  const m = currentTake.meta;
  $("t-detail").classList.remove("hidden");
  $("t-title").textContent = `${m.take_id} — ${label(m.emotion)} ${fmtInt(m.intensity)} (${m.kind})`;
  $("t-notes").value = m.annotations.notes || "";
  plot($("t-plot"), currentTake.t, cols(currentTake.q), m.joints, {segment: m.segment});
  const ev = m.event.present ? `az ${(m.event.azimuth_rad * 180 / Math.PI).toFixed(1)}°, el ${(m.event.elevation_rad * 180 / Math.PI).toFixed(1)}°` +
    (m.event.distance_m != null ? `, ${(+m.event.distance_m).toFixed(2)} m` : ", distance inconnue") + " (repère torse)" : "aucun événement";
  $("t-info").textContent = `session ${m.session_id} · ${m.n_samples} échantillons · ${m.sample_rate_hz} Hz · segment ${m.segment.map((x) => x.toFixed(2)).join("–")} s · calibration ${m.calibration_version}\n` +
    `guidées : ${m.guided_joints.join(", ")} · pilotées : ${m.driven_joints.join(", ") || "—"} · ${ev}` +
    (m.arm_take_id || m.arm_source ? `\nancienne prise enregistrée sur des bras rejoués${m.arm_take_id ? " : " + m.arm_take_id : " (geste généré)"} — seules les articulations ${m.kind === "head" ? "de la tête" : "des mains"} servent à l'entraînement` : "") +
    (currentTake.head_takes.length ? `\n${currentTake.head_takes.length} prise(s) de tête sur ce mouvement` : "") +
    ((currentTake.hand_takes || []).length ? `\n${currentTake.hand_takes.length} prise(s) de mains sur ce mouvement` : "") +
    (currentTake.replay ? `\nrelecture (guardian) : ${currentTake.replay.guardian.ok ? "acceptée" : "refusée " + JSON.stringify(currentTake.replay.guardian.violations)}` : "");
}

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
async function refreshTrain() {
  const d = await api("train");
  $("tr-jobs").innerHTML = d.jobs.map((j) => `<div class="mono">${j.id} · ${j.component} · ${j.state}${j.error ? " · " + j.error : ""}` +
    (j.progress.length ? ` · époque ${j.progress[j.progress.length - 1].epoch}, perte ${j.progress[j.progress.length - 1].train_loss.toFixed(4)}` : "") + "</div>").join("");
  $("tr-runs").innerHTML = d.runs.map((r) => {
    const m = r.metrics;
    return `<div class="draft"><b>${r.run}</b><div class="mono">exemples ${m.n_examples} (entraînement ${m.n_train} / validation ${m.n_val}, groupes ${m.n_train_groups}/${m.n_val_groups})\n` +
      `RMSE entraînement ${m.train_rmse_rad.toFixed(4)} rad · validation ${m.val_rmse_rad !== undefined ? m.val_rmse_rad.toFixed(4) : "—"} rad · référence (moyenne par étiquette) ${m.val_baseline_label_mean_rmse_rad !== undefined ? m.val_baseline_label_mean_rmse_rad.toFixed(4) : "—"} rad\n` +
      (m.val_duration_mae_s !== undefined ? `erreur de durée (val.) ${m.val_duration_mae_s.toFixed(3)} s\n` : "") +
      `guardian : ${m.guardian ? m.guardian.accepted + "/" + m.guardian.total + " trajectoires générées acceptées " + JSON.stringify(m.guardian.violations_by_code) : "—"}` +
      (m.warning ? "\n⚠ " + m.warning : "") + "</div></div>";
  }).join("");
}

// ---- génération -----------------------------------------------------------------
async function generate() {
  const e1 = $("g-emotion").value, e2 = $("g-emotion2").value, w = parseFloat($("g-mix").value);
  const body = {emotion: e1, intensity: parseFloat($("g-intensity").value), variation: parseFloat($("g-var").value), use_event: $("g-event").checked};
  if (e2 && e2 !== e1) body.mix = {[e1]: 1 - w, [e2]: w};
  const out = await api("generate", body);
  const tr = out.generated.trajectory;
  plot($("g-plot"), tr.t, cols(tr.q), tr.joints);
  animate($("g-plot"), 0, tr.t[tr.t.length - 1]);
  const mr = out.motion_report, cy = out.cycle;
  $("g-report").textContent = `geste : ${out.generated.duration_s.toFixed(2)} s · guardian : ${mr.ok ? "accepté" : "REFUSÉ"}\n` +
    (mr.violations.length ? JSON.stringify(mr.violations, null, 1) + "\n" : "") +
    (cy ? `cycle complet : ${cy.report.ok ? "accepté" : "REFUSÉ"} (${cy.duration_s.toFixed(2)} s) ${cy.report.violations.length ? JSON.stringify(cy.report.violations, null, 1) : ""}` : (out.cycle_error ? "cycle : " + out.cycle_error : "")) +
    (out.generated.notes.length ? "\nnotes : " + out.generated.notes.join(" ; ") : "");
  lastPlan = out.plan_id;
  $("g-exec").disabled = !(cy && cy.report.ok);
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
  // pas de confirmation (souhait de Joachim) : l'avertissement de la page (#a-alert) la remplace
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
  $("t-kind").onchange = loadTakes; $("t-refresh").onclick = loadTakes;
  $("t-play").onclick = () => {
    if (!currentTake) return;
    const m = currentTake.meta;
    if (currentTake.replay) {
      const r = currentTake.replay, off = r.offset;
      const t = r.t.map((x) => x + off);
      plot($("t-plot"), currentTake.t, cols(currentTake.q).concat(cols(r.q)), m.joints.concat(r.joints.map((j) => j + " (relecture)")), {segment: m.segment, dashed: r.joints.map((_, k) => m.joints.length + k)});
      // la relecture est tracée sur sa propre base de temps ; le curseur parcourt le segment
    } else plot($("t-plot"), currentTake.t, cols(currentTake.q), m.joints, {segment: m.segment});
    animate($("t-plot"), m.segment[0], m.segment[1]);
  };
  const upd = (body) => api("takes/" + currentTake.meta.take_id, body).then(() => { loadTakes(); openTake(currentTake.meta.take_id); });
  $("t-accept").onclick = () => upd({status: "accepted"});
  $("t-reject").onclick = () => upd({status: "rejected"});
  $("t-save-notes").onclick = () => upd({notes: $("t-notes").value});
  // entraînement
  const train = (c) => api("train", {component: c, epochs: parseInt($("tr-epochs").value), include_pending: $("tr-pending").checked}).then(refreshTrain);
  $("tr-arms").onclick = () => train("arms"); $("tr-head").onclick = () => train("head"); $("tr-hands").onclick = () => train("hands");
  setInterval(() => { if ($("tab-train").classList.contains("active")) refreshTrain(); }, 2000);
  // génération
  $("g-go").onclick = generate;
  $("g-exec").onclick = () => lastPlan && window.confirm(real() ? "Exécuter sur le ROBOT RÉEL ?" : "Exécuter sur le robot simulé ?") && api("execute", {plan_id: lastPlan, confirm: true});
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
