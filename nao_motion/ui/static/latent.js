"use strict";
// Generation tab, CVAE latent space (read only; nothing here moves the robot):
// - map of the z codes of the takes (2D PCA): one point per take, coloured by emotion. Click a
//   point: generate with that take's z and consigne (its playback by the model). Click elsewhere:
//   generate a new gesture from that z with the consigne of the tab.
// - sweep of one z dimension: the curves of each joint for z_i from -r to +r, overlaid.
(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const f2 = (x) => Number(x).toFixed(2);
  let group = "arms", lat = null, pickedZ = null, pickedTake = null;

  const emoColor = (e) => COLORS[Math.max(0, CFG.emotions.findIndex((x) => x.id === e)) % COLORS.length];
  // z <-> map coordinates (orthonormal PCA axes)
  const toZ = (x, y) => lat.pca.mean.map((m, i) => m + x * lat.pca.components[0][i] + y * lat.pca.components[1][i]);
  const toXY = (z) => [0, 1].map((k) => z.reduce((a, v, i) => a + (v - lat.pca.mean[i]) * lat.pca.components[k][i], 0));

  async function load() {
    $("lat-map").innerHTML = `<p class="hint">Chargement…</p>`;
    try { lat = await api("latent?component=" + group, undefined, {quiet: true}); }
    catch (e) { lat = {available: false, reason: e.message}; }
    pickedZ = null; pickedTake = null;
    if (!lat.available) {
      $("lat-map").innerHTML = `<p class="hint">${esc(lat.reason || "Indisponible")}</p>`;
      $("lat-legend").innerHTML = ""; $("lat-info").textContent = ""; $("lat-dim").innerHTML = ""; $("lat-sweep-out").innerHTML = "";
      $("lat-base").textContent = ""; return;
    }
    const v = lat.z_var_per_dim || [];
    $("lat-info").textContent = `modèle ${lat.run || "?"} · z de dimension ${lat.dim} · ${lat.z_active ?? "?"} active(s) · ${lat.take_ids.length} prises · axes : ${Math.round(lat.pca.explained[0] * 100)} % + ${Math.round(lat.pca.explained[1] * 100)} % de la variance des z` +
      (lat.z_active === 0 ? " · aucune dimension active : les prises ne montrent pas plusieurs façons de faire par consigne (ou β trop fort), la carte n'a pas de sens" : "");
    $("lat-dim").innerHTML = Array.from({length: lat.dim}, (_, i) => `<option value="${i}">z${i + 1}${v[i] !== undefined ? ` (variance ${f2(v[i])}${v[i] > 0.01 ? "" : ", inutilisée"})` : ""}</option>`).join("");
    // default dimension: the most used one
    if (v.length) $("lat-dim").value = String(v.indexOf(Math.max(...v)));
    drawMap(); baseInfo();
  }

  function drawMap() {
    const W = 520, H = 420, P = 30;
    const xy = lat.pca.xy, o0 = toXY(new Array(lat.dim).fill(0));
    let r = 2.5;  // prior N(0,1): ±2.5 covers almost every draw
    xy.forEach(([x, y]) => { r = Math.max(r, Math.abs(x - o0[0]) * 1.1, Math.abs(y - o0[1]) * 1.1); });
    const cx = o0[0], cy = o0[1];
    const X = (x) => W / 2 + (x - cx) / r * (W / 2 - P), Y = (y) => H / 2 - (y - cy) / r * (H / 2 - P);
    const k = (W / 2 - P) / r, ky = (H / 2 - P) / r;
    const o = [`<svg viewBox="0 0 ${W} ${H}" id="lat-svg">`,
      `<rect x="0" y="0" width="${W}" height="${H}" fill="transparent" id="lat-bg"/>`,
      `<ellipse cx="${X(cx)}" cy="${Y(cy)}" rx="${k}" ry="${ky}" fill="none" stroke="currentColor" stroke-opacity=".25" stroke-dasharray="4 4"/>`,
      `<ellipse cx="${X(cx)}" cy="${Y(cy)}" rx="${2 * k}" ry="${2 * ky}" fill="none" stroke="currentColor" stroke-opacity=".15" stroke-dasharray="4 4"/>`,
      `<line x1="${X(cx) - 7}" x2="${X(cx) + 7}" y1="${Y(cy)}" y2="${Y(cy)}" stroke="currentColor"/><line y1="${Y(cy) - 7}" y2="${Y(cy) + 7}" x1="${X(cx)}" x2="${X(cx)}" stroke="currentColor"/>`,
      `<text x="${X(cx) + 8}" y="${Y(cy) - 6}" font-size="11" fill="currentColor" opacity=".7">z = 0 (geste central)</text>`,
      `<text x="${X(cx) + k * 0.71 + 4}" y="${Y(cy) - ky * 0.71}" font-size="10" fill="currentColor" opacity=".5">1σ</text>`,
      `<text x="${X(cx) + 2 * k * 0.71 + 4}" y="${Y(cy) - 2 * ky * 0.71}" font-size="10" fill="currentColor" opacity=".5">2σ</text>`,
      `<text x="${W - 6}" y="${H - 6}" font-size="11" text-anchor="end" fill="currentColor" opacity=".7">axe 1 →</text>`,
      `<text x="6" y="14" font-size="11" fill="currentColor" opacity=".7">↑ axe 2</text>`];
    xy.forEach(([x, y], i) => {
      const e = lat.emotions[i], it = lat.intensities[i], ev = lat.event_present[i];
      o.push(`<circle class="pt${lat.take_ids[i] === pickedTake ? " sel" : ""}" data-i="${i}" cx="${X(x).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="${(3.5 + 4 * it).toFixed(1)}" fill="${emoColor(e)}" fill-opacity="${ev ? 0.55 : 0.9}"><title>${esc(lat.take_ids[i])}\n${esc(label(e))} · intensité ${fmtInt(it)}${ev ? " · avec événement" : ""}\nz = [${lat.mu[i].map(f2).join(", ")}]</title></circle>`);
    });
    if (pickedZ && !pickedTake) { const [x, y] = toXY(pickedZ); o.push(`<path d="M${X(x) - 6} ${Y(y) - 6}L${X(x) + 6} ${Y(y) + 6}M${X(x) + 6} ${Y(y) - 6}L${X(x) - 6} ${Y(y) + 6}" stroke="var(--bad)" stroke-width="2.5"/>`); }
    o.push("</svg>");
    $("lat-map").innerHTML = o.join("");
    const used = [...new Set(lat.emotions)];
    $("lat-legend").innerHTML = used.map((e) => `<span style="color:${emoColor(e)}">● ${esc(label(e))}</span>`).join(" ") +
      ` · taille = intensité · pâle = avec événement · cercles pointillés = 1σ et 2σ des tirages (T = 1)`;
    const svg = $("lat-svg");
    svg.onclick = (ev) => {
      const pt = ev.target.closest("circle.pt");
      if (pt) return pickTake(+pt.dataset.i);
      const b = svg.getBoundingClientRect();
      const sx = (ev.clientX - b.left) / b.width * W, sy = (ev.clientY - b.top) / b.height * H;
      const x = cx + (sx - W / 2) / (W / 2 - P) * r, y = cy - (sy - H / 2) / (H / 2 - P) * r;
      pickPoint(toZ(x, y));
    };
  }

  function baseInfo() {
    $("lat-base").textContent = pickedZ ? `autres dimensions au z choisi [${pickedZ.map(f2).join(", ")}]` : "autres dimensions à 0";
  }

  async function pickTake(i) {
    pickedTake = lat.take_ids[i]; pickedZ = lat.mu[i].slice();
    drawMap(); baseInfo();
    const extra = {latent: {[group]: pickedZ}, emotion: lat.emotions[i], intensity: lat.intensities[i], mix: null};
    if (lat.events && lat.events[i]) extra.event = lat.events[i]; else extra.use_event = false;
    $("lat-pick").textContent = `Prise ${pickedTake} (${label(lat.emotions[i])} · ${fmtInt(lat.intensities[i])}${lat.event_present[i] ? " · avec son événement" : ""}) rejouée par le modèle avec son z [${pickedZ.map(f2).join(", ")}]. ` +
      `Courbes et guardian ci-dessus. Pour la comparer à la prise enregistrée : onglet Prises, « modèle actif ».`;
    try { await generate(extra); } catch (e) {}
  }

  async function pickPoint(z) {
    pickedTake = null; pickedZ = z;
    drawMap(); baseInfo();
    $("lat-pick").textContent = `Nouveau geste depuis z = [${z.map(f2).join(", ")}] avec la consigne de l'onglet (émotion, intensité, événement). Courbes et guardian ci-dessus.`;
    try { await generate({latent: {[group]: z}}); } catch (e) {}
  }

  // diverging colours for the sweep: blue (−) → grey (0) → red (+)
  function sweepColor(t) {
    const a = [47, 98, 216], m = [150, 150, 150], b = [210, 60, 50];
    const [p, q, u] = t < 0.5 ? [a, m, t * 2] : [m, b, (t - 0.5) * 2];
    return `rgb(${p.map((v, i) => Math.round(v + (q[i] - v) * u)).join(",")})`;
  }

  async function sweep() {
    if (!lat || !lat.available) return;
    const body = {...genConsigne(), component: group, dim: parseInt($("lat-dim").value), range: parseFloat($("lat-range").value),
                  steps: parseInt($("lat-steps").value)};
    if (pickedZ) body.z = pickedZ;
    $("lat-sweep-out").innerHTML = `<p class="hint">Calcul…</p>`;
    let d;
    try { d = await api("latent_sweep", body); } catch (e) { $("lat-sweep-out").innerHTML = `<p class="hint">${esc(e.message)}</p>`; return; }
    // joints that change the most first; still joints are left out
    const order = d.joints.map((j, i) => i).sort((a, b) => d.spread_per_joint[b] - d.spread_per_joint[a]);
    const shown = order.filter((i) => d.spread_per_joint[i] > 0.005).slice(0, 10);
    const still = order.filter((i) => !shown.includes(i)).map((i) => d.joints[i]);
    const n = d.values.length, Tmax = Math.max(...d.durations);
    const W = 300, H = 150, P = 22;
    const cards = shown.map((j) => {
      let lo = Infinity, hi = -Infinity;
      d.curves.forEach((c) => c.forEach((row) => { lo = Math.min(lo, row[j]); hi = Math.max(hi, row[j]); }));
      if (hi - lo < 0.2) { const m = (hi + lo) / 2; lo = m - 0.1; hi = m + 0.1; }  // a tiny effect stays visibly tiny
      const X = (t) => P + t / Tmax * (W - 2 * P), Y = (v) => H - P - (v - lo) / (hi - lo) * (H - 2 * P);
      const lines = d.curves.map((c, k) => `<polyline fill="none" stroke="${sweepColor(n > 1 ? k / (n - 1) : 0.5)}" stroke-width="${k === Math.floor(n / 2) && n % 2 ? 2.2 : 1.6}" points="${c.map((row, i) => `${X(d.phase[i] * d.durations[k]).toFixed(1)},${Y(row[j]).toFixed(1)}`).join(" ")}"/>`).join("");
      return `<div><div class="jn"><b>${esc(d.joints[j])}</b> · écart ${f2(d.spread_per_joint[j])} rad</div><svg viewBox="0 0 ${W} ${H}">` +
        `<text x="2" y="${P}" font-size="10" fill="currentColor">${f2(hi)}</text><text x="2" y="${H - P}" font-size="10" fill="currentColor">${f2(lo)}</text>` +
        `<text x="${W - P}" y="${H - 6}" font-size="10" text-anchor="end" fill="currentColor">${Tmax.toFixed(1)} s</text>${lines}</svg></div>`;
    });
    const leg = d.values.map((v, k) => `<span style="color:${sweepColor(n > 1 ? k / (n - 1) : 0.5)}">━ z${d.dim + 1} = ${v > 0 ? "+" : ""}${f2(v)} (${d.durations[k].toFixed(2)} s)</span>`).join(" ");
    $("lat-sweep-out").innerHTML = `<div class="legend">${leg}</div>` +
      (cards.length ? `<div class="lat-sw">${cards.join("")}</div>` : `<p class="hint">Cette dimension ne change aucune articulation de plus de 0,005 rad : elle est inutilisée.</p>`) +
      (still.length && cards.length ? `<div class="legend">Peu ou pas touchées par z${d.dim + 1} : ${still.map(esc).join(", ")}</div>` : "");
  }

  function init() {
    document.querySelectorAll("[data-lat-group]").forEach((b) => b.onclick = () => {
      group = b.dataset.latGroup;
      document.querySelectorAll("[data-lat-group]").forEach((x) => x.classList.toggle("sel", x === b));
      load();
    });
    $("lat-reload").onclick = load;
    $("lat-sweep").onclick = sweep;
    $("g-latent").addEventListener("toggle", () => { if ($("g-latent").open && !lat) load(); });
  }

  window.LATENT = {
    reload: load,
    // after any generation: keep the map, note the z of this group when it was drawn at random
    onGenerated(out) {
      const z = out && out.generated && out.generated.inputs.latent && out.generated.inputs.latent[group];
      if (!z || !lat || !lat.available || !$("g-latent").open) return;
      pickedZ = z.slice(); baseInfo();
      if (!pickedTake || JSON.stringify(lat.mu[lat.take_ids.indexOf(pickedTake)]) !== JSON.stringify(z)) { pickedTake = null; drawMap(); }
    },
  };
  const wait = setInterval(() => { if (typeof CFG !== "undefined" && CFG && CFG.emotions) { clearInterval(wait); init(); } }, 50);
})();
