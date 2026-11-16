"use strict";
// Coverage map of the recording tabs (1 · Arms, 2 · Head, 3 · Hands), above the NAO figure:
// accepted real takes of the current group (same count as the Takes tab), one row per
// emotion, one column per intensity slice (grouped under the draw's intensity bins). The
// more takes, the darker the cell; each cell shows its count. The cell the take being set
// up will fill is outlined. Display only: nothing here moves the robot, the map is read
// from GET /api/coverage.
(function () {
  const KINDS = {arms: "a", head: "h", hands: "m"};
  const box = document.getElementById("coverage");
  if (!box) return;
  let lastKey = "", lastFetch = 0, pending = false, data = null;

  const activeTab = () => { const t = document.querySelector("section.tab.active"); return t ? t.id.slice(4) : ""; };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const emoLabel = (id) => ((typeof CFG !== "undefined" && CFG && CFG.emotions.find((e) => e.id === id)) || {label: id}).label;
  const num = (x) => String(x).replace(".", ",");

  function render(kind) {
    const d = data;
    if (!d || d.kind !== kind) return;
    const [cr, cc] = d.current || [-1, -1];
    const bands = [];
    d.columns.forEach((c) => {
      const b = bands[bands.length - 1];
      if (b && b.band === c.band) b.n++; else bands.push({band: c.band, n: 1});
    });
    let h = `<table class="cov-grid"><colgroup><col class="cov-c-emo"></colgroup><thead><tr><th></th>` +
      bands.map((b) => `<th colspan="${b.n}" class="cov-band">${esc(d.bands[b.band])}</th>`).join("") + "</tr></thead><tbody>";
    d.emotions.forEach((e, i) => {
      h += `<tr><th class="cov-emo${i === cr ? " cov-cur" : ""}">${esc(emoLabel(e))}</th>`;
      d.counts[i].forEach((c, j) => {
        const f = d.max > 0 && c > 0 ? 0.2 + 0.8 * Math.sqrt(c / d.max) : 0;
        const col = d.columns[j];
        h += `<td class="${c > 0 ? "has" : ""}${f > 0.6 ? " dark" : ""}${i === cr && j === cc ? " cur" : ""}"` +
          (f ? ` style="--f:${Math.round(f * 100)}%"` : "") +
          ` title="${esc(emoLabel(e))}, intensity ${num(col.lo)}–${num(col.hi)}: ${c} take(s)">${c || ""}</td>`;
      });
      h += "</tr>";
    });
    const last = d.columns.length - 1;
    const ticks = d.columns.map((c, j) => j === 0 ? `<td>${num(c.lo)}</td>` : j === last ? `<td class="cov-end">${num(c.hi)}</td>` : "<td></td>").join("");
    h += `</tbody><tfoot><tr><th></th>${ticks}</tr></tfoot></table>`;
    const cur = d.current ? d.counts[cr][cc] : null;
    h += `<div class="cov-foot"><span>${d.accepted} accepted</span>` +
      (cur !== null ? `<span class="cov-next">▢ this take's cell: ${cur}</span>` : "") + "</div>" +
      (d.simulated ? `<div class="cov-sim">${d.simulated} simulated take(s) not counted</div>` : "");
    box.querySelector(".cov-body").innerHTML = h;
  }

  async function refresh(kind, p) {
    if (pending) return;
    pending = true;
    lastFetch = performance.now();
    const q = new URLSearchParams({kind, emotion: $(p + "-emotion").value, intensity: $(p + "-intensity").value});
    try { data = await api("coverage?" + q, undefined, {quiet: true, timeoutMs: 4000}); render(kind); }
    catch (e) { /* the map is only informative: keep the last one */ }
    finally { pending = false; }
  }

  function tick() {
    const kind = activeTab(), p = KINDS[kind];
    box.classList.toggle("hidden", !p);
    if (!p) return;
    const st = typeof STATUS !== "undefined" && STATUS ? STATUS : {};
    const draft = document.getElementById(p + "-draft");
    const key = [kind, $(p + "-emotion").value, $(p + "-intensity").value, (st.recording || {}).state,
      draft && draft.classList.contains("hidden")].join("|");
    if (key !== lastKey || performance.now() - lastFetch > 20000) { lastKey = key; refresh(kind, p); }
  }

  box.innerHTML = `<div class="cov-title" title="Accepted takes of this group, as in the Takes tab: rows = emotion, columns = intensity.">Accepted takes · emotion × intensity</div><div class="cov-body"></div>`;
  window.COVERAGE = {refresh: () => { lastKey = ""; tick(); }};
  setInterval(tick, 500);
  tick();
})();
