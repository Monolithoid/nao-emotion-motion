"use strict";
const $ = (id) => document.getElementById(id);
let CFG = null, STATUS = null, lastPlan = null, currentTake = null;
const COLORS = ["#2f62d8","#d8572f","#1d8a4b","#9b3fc4","#c49b1f","#1f9bc4","#c41f6b","#6b7a1f","#555","#e08a00","#0aa","#a33"];

async function api(path, body) {
  const opt = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
  const r = await fetch("/api/" + path, opt);
  const d = await r.json();
  if (!r.ok) { const msg = d.error + (d.report ? "\n" + JSON.stringify(d.report.violations, null, 1) : ""); alert(msg); throw new Error(msg); }
  return d;
}
const real = () => STATUS && STATUS.real_robot;
const confirmReal = (msg) => !real() || window.confirm(msg);
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
async function refreshStatus() {
  try { STATUS = await api("status"); } catch (e) { return; }
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
    r.state === "done" ? `Terminé (${r.elapsed.toFixed(1)} s)${r.complete === false ? " — incomplet" : ""}` : r.state + (r.error ? " : " + r.error : "");
  if (r.state !== "none") (r.kind === "head" ? $("h-rec") : $("a-rec")).textContent = txt;
  if (r.state === "done") showDraft(r.kind);
  $("r-status").textContent = JSON.stringify(STATUS, null, 1);
  if (window.FOV) window.FOV.sync(STATUS.event);
}

let draftShown = null;
function showDraft(kind) {
  if (draftShown === kind) return;
  draftShown = kind;
  $(kind === "head" ? "h-draft" : "a-draft").classList.remove("hidden");
  const plotEl = $(kind === "head" ? "h-plot" : "a-plot");
  api("record/draft").then((d) => {
    plot(plotEl, d.t, cols(d.q), d.joints, {segment: d.segment});
    animate(plotEl, d.segment[0], d.segment[1]);
  });
}

async function saveDraft(kind, status) {
  const notes = $(kind === "head" ? "h-notes" : "a-notes").value;
  const r = await api("record/save", {status, notes});
  draftShown = null;
  $(kind === "head" ? "h-draft" : "a-draft").classList.add("hidden");
  const take = await api("takes/" + r.take_id);
  const box = $(kind === "head" ? "h-rec" : "a-rec");
  box.textContent = `Prise ${r.take_id} : ${status}`;
  await loadTakes();
  return take;
}

// ---- bras ----------------------------------------------------------------------
function updateConsigne() {
  $("a-int-val").textContent = fmtInt($("a-intensity").value);
  $("a-consigne").textContent = `${label($("a-emotion").value)} — ${fmtInt($("a-intensity").value)}` + ($("a-intention").value ? ` (${$("a-intention").value})` : "");
}

// ---- prises ----------------------------------------------------------------------
let TAKES = [];
async function loadTakes() {
  TAKES = await api("takes" + ($("t-kind").value ? "?kind=" + $("t-kind").value : ""));
  const all = await api("takes");
  const armSel = $("h-arm"), prev = armSel.value;
  armSel.innerHTML = all.filter((m) => m.kind === "arms" && m.status !== "rejected")
    .map((m) => `<option value="${m.take_id}">${label(m.emotion)} ${fmtInt(m.intensity)} · ${m.take_id} · ${m.status}</option>`).join("");
  if (prev) armSel.value = prev;
  $("t-table").innerHTML = "<tr><th>Prise</th><th>Type</th><th>Émotion</th><th>Int.</th><th>Durée</th><th>Événement</th><th>Statut</th><th>Notes</th></tr>" +
    TAKES.map((m) => `<tr class="clickable" data-id="${m.take_id}"><td>${m.take_id}${m.synthetic ? " <span class='tag'>synth</span>" : ""}</td><td>${m.kind}</td><td>${label(m.emotion)}</td>` +
      `<td>${fmtInt(m.intensity)}</td><td>${m.duration_s.toFixed(1)} s</td><td>${m.event.present ? (m.event.azimuth_rad * 180 / Math.PI).toFixed(0) + "°" : "aucun"}</td>` +
      `<td class="${m.status}">${m.status}</td><td>${(m.annotations.notes || "")}</td></tr>`).join("");
  document.querySelectorAll("#t-table tr.clickable").forEach((tr) => tr.onclick = () => openTake(tr.dataset.id));
  const st = await api("stats");
  let h = "";
  for (const kind of ["arms", "head"]) {
    h += `<h4>${kind === "arms" ? "Bras" : "Tête"}</h4><table><tr><th></th>${st.bins.map((b) => `<th>${b}</th>`).join("")}</tr>`;
    for (const e of CFG.emotions) {
      h += `<tr><td>${e.label}</td>` + st.bins.map((b) => { const c = st.table[kind][e.id][b] || {}; const tot = Object.values(c).reduce((a, x) => a + x, 0); return `<td>${c.accepted || 0} / ${tot}</td>`; }).join("") + "</tr>";
    }
    h += "</table>";
  }
  $("stats").innerHTML = h;
  showArmForHead();
}

async function openTake(id) {
  currentTake = await api("takes/" + id);
  const m = currentTake.meta;
  $("t-detail").classList.remove("hidden");
  $("t-title").textContent = `${m.take_id} — ${label(m.emotion)} ${fmtInt(m.intensity)} (${m.kind})`;
  $("t-notes").value = m.annotations.notes || "";
  plot($("t-plot"), currentTake.t, cols(currentTake.q), m.joints, {segment: m.segment});
  const ev = m.event.present ? `az ${(m.event.azimuth_rad * 180 / Math.PI).toFixed(1)}°, el ${(m.event.elevation_rad * 180 / Math.PI).toFixed(1)}° (repère torse)` : "aucun événement";
  $("t-info").textContent = `session ${m.session_id} · ${m.n_samples} échantillons · ${m.sample_rate_hz} Hz · segment ${m.segment.map((x) => x.toFixed(2)).join("–")} s · calibration ${m.calibration_version}\n` +
    `guidées : ${m.guided_joints.join(", ")} · pilotées : ${m.driven_joints.join(", ") || "—"} · ${ev}` +
    (m.arm_take_id ? `\nprise de bras rejouée : ${m.arm_take_id}` : "") +
    (currentTake.head_takes.length ? `\n${currentTake.head_takes.length} prise(s) de tête sur ce mouvement` : "") +
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

async function showArmForHead() {
  const id = $("h-arm").value;
  if (!id) { $("h-arm-plot").innerHTML = ""; return; }
  const tk = await api("takes/" + id);
  plot($("h-arm-plot"), tk.t, cols(tk.q), tk.meta.joints, {segment: tk.meta.segment});
  $("h-arm-info").textContent = `${tk.head_takes.length} contexte(s) de tête déjà enregistré(s)`;
  if (document.querySelector("input[name=hsrc]:checked").value === "take")
    $("h-consigne").textContent = `${label(tk.meta.emotion)} — ${fmtInt(tk.meta.intensity)} (démonstration rejouée)`;
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
  $("a-emotion").innerHTML = opts; $("g-emotion").innerHTML = opts; $("h-emotion").innerHTML = opts;
  const hcons = () => { $("h-int-val").textContent = fmtInt($("h-intensity").value);
    if (document.querySelector("input[name=hsrc]:checked").value === "generated")
      $("h-consigne").textContent = `${label($("h-emotion").value)} — ${fmtInt($("h-intensity").value)} (geste généré)`; else showArmForHead(); };
  ["h-emotion", "h-intensity"].forEach((id) => $(id).oninput = hcons);
  document.querySelectorAll("input[name=hsrc]").forEach((r) => r.onchange = hcons);
  hcons(); $("g-emotion2").innerHTML += opts;
  $("a-intention").innerHTML += CFG.intentions.map((i) => `<option>${i}</option>`).join("");
  ["a-emotion", "a-intensity", "a-intention"].forEach((id) => $(id).oninput = updateConsigne);
  $("g-intensity").oninput = () => $("g-int-val").textContent = fmtInt($("g-intensity").value);
  updateConsigne(); $("g-int-val").textContent = fmtInt(0.7);
  document.querySelectorAll("nav button").forEach((b) => b.onclick = () => {
    document.querySelectorAll("nav button, .tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active"); $("tab-" + b.dataset.tab).classList.add("active");
    if (b.dataset.tab === "train") refreshTrain();
    if (b.dataset.tab === "takes" || b.dataset.tab === "head") loadTakes();
  });
  $("stop").onclick = () => api("stop", {});
  $("armed").onchange = async (e) => {
    const on = e.target.checked;
    if (on && !window.confirm("Autoriser les mouvements motorisés ? Chaque mouvement restera déclenché par un bouton explicite.")) { e.target.checked = false; return; }
    await api("arm", {armed: on, confirm: on}).catch(() => (e.target.checked = false));
  };
  // bras
  $("a-prepare").onclick = () => confirmReal("Les bras vont être RELÂCHÉS. Les tenez-vous ?") &&
    api("record/prepare", {kind: "arms", emotion: $("a-emotion").value, intensity: parseFloat($("a-intensity").value), intention: $("a-intention").value, confirm: true}).then(() => { draftShown = null; });
  $("a-start").onclick = () => api("record/start", {});
  $("a-stop").onclick = () => api("record/stop", {});
  $("a-hold").onclick = () => api("hold", {groups: ["left_arm", "right_arm"]});
  $("a-accept").onclick = () => saveDraft("arms", "accepted");
  $("a-pending").onclick = () => saveDraft("arms", "pending");
  $("a-reject").onclick = () => saveDraft("arms", "rejected");
  $("a-retry").onclick = async () => { await saveDraft("arms", "rejected"); $("a-prepare").onclick(); };
  // tête
  $("h-arm").onchange = showArmForHead;
  $("h-prepare").onclick = () => confirmReal("Les bras vont se déplacer (motorisés) vers le début de la relecture, puis la tête sera RELÂCHÉE. Continuer ?") &&
    api("record/prepare", document.querySelector("input[name=hsrc]:checked").value === "generated"
      ? {kind: "head", confirm: true, generated: {emotion: $("h-emotion").value, intensity: parseFloat($("h-intensity").value),
          variation: parseFloat($("h-var").value), seed: parseInt($("h-seed").value)}}
      : {kind: "head", arm_take_id: $("h-arm").value, confirm: true}).then(() => { draftShown = null; });
  $("h-start").onclick = () => api("record/start", {});
  $("h-stop").onclick = () => api("stop", {});
  $("h-hold").onclick = () => api("hold", {groups: ["head"]});
  $("h-accept").onclick = () => saveDraft("head", "accepted");
  $("h-pending").onclick = () => saveDraft("head", "pending");
  $("h-reject").onclick = () => saveDraft("head", "rejected");
  $("h-retry").onclick = async () => { await saveDraft("head", "rejected"); $("h-prepare").onclick(); };
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
  $("t-use-head").onclick = () => { if (currentTake.meta.kind !== "arms") return alert("Choisir une prise de bras"); document.querySelector("nav button[data-tab=head]").click(); setTimeout(() => { $("h-arm").value = currentTake.meta.take_id; showArmForHead(); }, 300); };
  // entraînement
  const train = (c) => api("train", {component: c, epochs: parseInt($("tr-epochs").value), include_pending: $("tr-pending").checked}).then(refreshTrain);
  $("tr-arms").onclick = () => train("arms"); $("tr-head").onclick = () => train("head");
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
  $("r-connect-robot").onclick = async () => {
    const v = {host: $("r-host").value.trim(), port: parseInt($("r-port").value), bhost: $("r-bhost").value.trim(), bport: parseInt($("r-bport").value)};
    if (!v.host) return alert("Indiquer l'adresse du robot (IP ou nom DNS)");
    await api("connect", {mode: "robot", robot_host: v.host, robot_port: v.port, bridge_host: v.bhost, bridge_port: v.bport});
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
