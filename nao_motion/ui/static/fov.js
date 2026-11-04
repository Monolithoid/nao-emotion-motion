"use strict";
// Champ de vision : l'événement vu par le robot (repère torse).
// Abscisse = azimut (gauche du robot à gauche), ordonnée = élévation,
// taille du point = proximité (plus gros = plus proche). Un clic place le point.
// Le serveur garde UN événement courant : chaque champ l'affiche et le modifie.
(function () {
  const DEG = 180 / Math.PI;
  const W = 640, H = 360, P = 34, B = H - 80;  // B : bas du champ ; dessous, la légende des tailles
  const widgets = [];
  let R = null;           // plages (config)
  let lastEdit = 0;       // évite d'écraser une saisie en cours par la synchronisation

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const prox = (d) => 1 / (1 + d);

  function ranges() {
    const e = CFG.events;
    const az = [Math.max(-180, e.azimuth_deg[0] - 30), Math.min(180, e.azimuth_deg[1] + 30)];
    const el = [Math.max(-90, e.elevation_deg[0] - 15), Math.min(90, e.elevation_deg[1] + 15)];
    return {az, el, draw: e, dist: e.distance_m};
  }

  function radius(d) {
    if (d == null) return 11;
    const [d0, d1] = R.dist, p0 = prox(d1), p1 = prox(d0);
    return 5 + 21 * clamp((prox(d) - p0) / (p1 - p0), 0, 1);
  }

  function build(el) {
    const id = el.id;
    el.classList.add("fov");
    el.innerHTML =
      `<div class="row">
        <label><input type="checkbox" id="${id}-present"> événement présent</label>
        <label>azimut <input type="number" id="${id}-az" step="5" class="num">°</label>
        <label>élévation <input type="number" id="${id}-el" step="5" class="num">°</label>
        <label>distance <input type="number" id="${id}-d" step="0.1" min="0.05" class="num"> m</label>
        <label><input type="checkbox" id="${id}-dunk"> distance inconnue</label>
        <span id="${id}-txt" class="hint"></span>
      </div>
      <svg id="${id}-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Champ de vision du robot"></svg>`;
    const w = {el, id, ev: {present: false}};
    const X = (az) => P + (R.az[1] - az) / (R.az[1] - R.az[0]) * (W - 2 * P);   // gauche du robot à gauche
    const Y = (e) => P + (R.el[1] - e) / (R.el[1] - R.el[0]) * (B - P);
    w.X = X; w.Y = Y;
    const inp = (k) => document.getElementById(`${id}-${k}`);
    w.inp = inp;
    const send = () => {
      lastEdit = Date.now();
      const present = inp("present").checked;
      const body = present ? {present: true, azimuth: parseFloat(inp("az").value) || 0, elevation: parseFloat(inp("el").value) || 0,
        unit: "deg", frame: "torso", source: "interface",
        distance_m: inp("dunk").checked ? null : (parseFloat(inp("d").value) || null)} : {present: false, source: "interface"};
      api("event", body).then((ev) => { for (const o of widgets) show(o, ev); });
    };
    ["present", "az", "el", "d", "dunk"].forEach((k) => inp(k).onchange = send);
    inp("svg").onclick = (e) => {
      const r = inp("svg").getBoundingClientRect();
      const sx = (e.clientX - r.left) / r.width * W, sy = (e.clientY - r.top) / r.height * H;
      const az = R.az[1] - (sx - P) / (W - 2 * P) * (R.az[1] - R.az[0]);
      const elv = R.el[1] - (sy - P) / (B - P) * (R.el[1] - R.el[0]);
      inp("present").checked = true;
      inp("az").value = Math.round(clamp(az, R.az[0], R.az[1]));
      inp("el").value = Math.round(clamp(elv, R.el[0], R.el[1]));
      send();
    };
    widgets.push(w);
    return w;
  }

  function grid(w) {
    const {X, Y} = w;
    let s = `<rect x="${P}" y="${P}" width="${W - 2 * P}" height="${B - P}" rx="8" fill="var(--card)" stroke="var(--line)"/>`;
    // zone du tirage aléatoire
    const a = R.draw.azimuth_deg, e = R.draw.elevation_deg;
    s += `<rect x="${X(a[1])}" y="${Y(e[1])}" width="${X(a[0]) - X(a[1])}" height="${Y(e[0]) - Y(e[1])}" fill="var(--accent)" fill-opacity=".06" stroke="var(--accent)" stroke-opacity=".35" stroke-dasharray="4 4"/>`;
    for (let az = Math.ceil(R.az[0] / 30) * 30; az <= R.az[1]; az += 30)
      s += `<line x1="${X(az)}" x2="${X(az)}" y1="${P}" y2="${B}" stroke="var(--line)"${az === 0 ? ' stroke-width="1.5"' : ""}/>` +
        `<text x="${X(az)}" y="${B + 14}" text-anchor="middle" font-size="11" fill="var(--muted)">${az}°</text>`;
    for (let el = Math.ceil(R.el[0] / 15) * 15; el <= R.el[1]; el += 15)
      s += `<line x1="${P}" x2="${W - P}" y1="${Y(el)}" y2="${Y(el)}" stroke="var(--line)"${el === 0 ? ' stroke-width="1.5"' : ""}/>` +
        `<text x="${P - 4}" y="${Y(el) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${el}°</text>`;
    s += `<text x="${P}" y="${P - 10}" font-size="11" fill="var(--muted)">← gauche du robot</text>` +
      `<text x="${W - P}" y="${P - 10}" text-anchor="end" font-size="11" fill="var(--muted)">droite du robot →</text>` +
      `<text x="${W / 2}" y="${P - 10}" text-anchor="middle" font-size="11" fill="var(--muted)">devant</text>`;
    // légende des tailles
    const ly = H - 30, dm = Math.sqrt(R.dist[0] * R.dist[1]);
    s += `<text x="${P}" y="${ly + 4}" font-size="11" fill="var(--muted)">Taille du point = proximité :</text>`;
    [[R.dist[1], 200], [dm, 270], [R.dist[0], 350]].forEach(([d, x]) => {
      s += `<circle cx="${x}" cy="${ly}" r="${radius(d)}" fill="var(--bad)" fill-opacity=".35"/>` +
        `<text x="${x + radius(d) + 5}" y="${ly + 4}" font-size="11" fill="var(--muted)">${d.toFixed(d < 1 ? 1 : 1)} m</text>`;
    });
    s += `<circle cx="${470}" cy="${ly}" r="${radius(null)}" fill="none" stroke="var(--bad)" stroke-width="2" stroke-dasharray="3 3"/>` +
      `<text x="${470 + radius(null) + 5}" y="${ly + 4}" font-size="11" fill="var(--muted)">distance inconnue</text>`;
    return s;
  }

  function show(w, ev) {
    w.ev = ev;
    const inp = w.inp, focused = w.el.contains(document.activeElement) && document.activeElement.tagName === "INPUT";
    if (!focused) {
      inp("present").checked = !!ev.present;
      if (ev.present) {
        inp("az").value = Math.round(ev.azimuth_rad * DEG);
        inp("el").value = Math.round(ev.elevation_rad * DEG);
        inp("dunk").checked = ev.distance_m == null;
        if (ev.distance_m != null) inp("d").value = (+ev.distance_m).toFixed(2);
      }
    }
    let s = grid(w);
    if (ev.present) {
      const x = w.X(clamp(ev.azimuth_rad * DEG, R.az[0], R.az[1])), y = w.Y(clamp(ev.elevation_rad * DEG, R.el[0], R.el[1]));
      const r = radius(ev.distance_m);
      s += ev.distance_m == null
        ? `<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="var(--bad)" stroke-width="2" stroke-dasharray="3 3"/>`
        : `<circle cx="${x}" cy="${y}" r="${r}" fill="var(--bad)" fill-opacity=".75" stroke="var(--bad)"/>`;
    }
    inp("svg").innerHTML = s;
    inp("txt").textContent = ev.present
      ? `${ev.source} · ${ev.distance_m == null ? "distance inconnue" : (+ev.distance_m).toFixed(2) + " m"}`
      : `aucun événement (${ev.source || "—"})`;
  }

  // Tirage : émotion, intensité, angle et distance (log-uniforme : autant de proches que de lointains).
  function draw(allowNone) {
    const e = CFG.events, u = (a, b) => a + Math.random() * (b - a);
    const emotion = CFG.emotions[Math.floor(Math.random() * CFG.emotions.length)].id;
    const intensity = Math.round(u(0.1, 1) * 20) / 20;
    const ev = allowNone && Math.random() < 0.2 ? {present: false, source: "tirage"} : {
      present: true, unit: "deg", frame: "torso", source: "tirage",
      azimuth: Math.round(u(e.azimuth_deg[0], e.azimuth_deg[1])), elevation: Math.round(u(e.elevation_deg[0], e.elevation_deg[1])),
      distance_m: Math.round(Math.exp(u(Math.log(e.distance_m[0]), Math.log(e.distance_m[1]))) * 100) / 100};
    return {emotion, intensity, ev};
  }

  async function applyDraw(d) {
    lastEdit = Date.now();
    const ev = await api("event", d.ev);
    for (const o of widgets) show(o, ev);
  }

  function setup() {
    if (typeof CFG === "undefined" || !CFG || !CFG.events) return setTimeout(setup, 100);
    R = ranges();
    for (const id of ["h-fov", "g-fov"]) { const el = $(id); if (el) show(build(el), {present: false}); }
    api("event").then((ev) => widgets.forEach((w) => show(w, ev)));
    $("a-random").onclick = () => {
      const d = draw(false);
      $("a-emotion").value = d.emotion; $("a-intensity").value = d.intensity; updateConsigne();
    };
    $("h-random").onclick = async () => {
      const d = draw($("h-random-none").checked);
      document.querySelector("input[name=hsrc][value=generated]").checked = true;
      $("h-emotion").value = d.emotion; $("h-intensity").value = d.intensity;
      await applyDraw(d);
      $("h-emotion").oninput();
    };
    $("g-random").onclick = async () => {
      const d = draw(false);
      $("g-emotion").value = d.emotion; $("g-emotion2").value = ""; $("g-intensity").value = d.intensity;
      $("g-intensity").oninput(); $("g-event").checked = true;
      await applyDraw(d);
    };
  }

  window.FOV = {
    sync(ev) {
      if (!R || !ev || Date.now() - lastEdit < 1000) return;
      for (const w of widgets) {
        const same = JSON.stringify([w.ev.present, w.ev.azimuth_rad, w.ev.elevation_rad, w.ev.distance_m, w.ev.source]) ===
          JSON.stringify([ev.present, ev.azimuth_rad, ev.elevation_rad, ev.distance_m, ev.source]);
        if (!same) show(w, ev);
      }
    },
  };
  setup();
})();
