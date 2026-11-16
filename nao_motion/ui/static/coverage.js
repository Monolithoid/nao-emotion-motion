"use strict";
// Coverage map of the recording tabs (1 · Arms, 2 · Head, 3 · Hands), above the NAO figure:
// accepted real takes of the current group (same count as the Takes tab), one cell per
// emotion x intensity (rows) and event zone (columns: robot's left first, then front, then
// right; ↑/↓ high or low, n/f near or far; ∅ no event). The more takes, the darker the cell.
// The cell(s) the take being set up will fill is outlined. Display only: nothing here moves
// the robot, the map is read from GET /api/coverage.
(function () {
  const KINDS = {arms: "a", head: "h", hands: "m"};
  const box = document.getElementById("coverage");
  if (!box) return;
  let lastKey = "", lastFetch = 0, pending = false, data = null;

  const activeTab = () => { const t = document.querySelector("section.tab.active"); return t ? t.id.slice(4) : ""; };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const emoLabel = (id) => ((typeof CFG !== "undefined" && CFG && CFG.emotions.find((e) => e.id === id)) || {label: id}).label;
  const fmt = (x) => String(Math.round(x * 10) / 10);

  function render(kind) {
    const d = data;
    if (!d || d.kind !== kind) return;
    const cur = new Set(d.current.map(([r, c]) => r + ":" + c));
    const curRows = new Set(d.current.map(([r]) => r));
    const nz = d.zones.length;
    // header: azimuth groups (robot's left first), then the short zone codes
    let groups = [];
    d.zones.forEach((z, j) => {
      const key = z.az === null ? "none" : "az" + z.az;
      if (groups.length && groups[groups.length - 1].key === key) groups[groups.length - 1].n++;
      else groups.push({key, n: 1, label: z.az === null ? "" : z.az_label});
    });
    let h = `<table class="cov-grid"><colgroup><col class="cov-c-emo"><col class="cov-c-int"></colgroup>` +
      `<thead><tr><th colspan="2"></th>` +
      groups.map((g) => `<th colspan="${g.n}" class="cov-az">${esc(g.label)}</th>`).join("") + `</tr><tr><th colspan="2"></th>` +
      d.zones.map((z) => `<th class="cov-z" title="${esc(z.label)}">${esc(z.short)}</th>`).join("") + "</tr></thead><tbody>";
    const bars = ["▂", "▄", "▆", "█"];
    d.rows.forEach((r, i) => {
      const first = r.int_index === 0;
      const emoCur = d.current.some(([k]) => d.rows[k].emotion === r.emotion);
      h += `<tr class="${first ? "cov-first" : ""}${curRows.has(i) ? " cov-cur-row" : ""}">` +
        (first ? `<th class="cov-emo${emoCur ? " cov-cur" : ""}" rowspan="${d.n_int}">${esc(emoLabel(r.emotion))}</th>` : "") +
        `<th class="cov-int" title="intensity ${esc(r.intensity)}">${bars[Math.min(r.int_index, 3)]}</th>`;
      d.counts[i].forEach((c, j) => {
        const f = d.max > 0 && c > 0 ? 0.18 + 0.82 * Math.sqrt(c / d.max) : 0;
        const style = f ? ` style="--f:${Math.round(f * 100)}%"` : "";
        h += `<td class="${c > 0 ? "has" : ""}${cur.has(i + ":" + j) ? " cur" : ""}"${style}` +
          ` title="${esc(emoLabel(r.emotion))}, intensity ${esc(r.intensity)}, ${esc(d.zones[j].label)}: ${fmt(c)} take(s)"></td>`;
      });
      h += "</tr>";
    });
    h += "</tbody></table>";
    const curCount = d.current.reduce((a, [r, c]) => a + d.counts[r][c], 0);
    h += `<div class="cov-foot"><span>${d.accepted} accepted` + (d.max ? `, up to ${fmt(d.max)} per cell` : "") + `</span>` +
      (d.current.length ? `<span class="cov-next">▢ this take's cell${d.current.length > 1 ? "s" : ""}: ${fmt(curCount)}</span>` : "") + "</div>" +
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
    const key = [kind, $(p + "-emotion").value, $(p + "-intensity").value, JSON.stringify(st.event || {}),
      (st.recording || {}).state, draft && draft.classList.contains("hidden")].join("|");
    if (key !== lastKey || performance.now() - lastFetch > 20000) { lastKey = key; refresh(kind, p); }
  }

  box.innerHTML = `<div class="cov-title" title="Accepted takes of this group, as in the Takes tab. Columns: event zone (robot's left first; ↑ high, ↓ low; n near, f far; ∅ no event). Rows: emotion and intensity.">Accepted takes per cell</div><div class="cov-body"></div>`;
  window.COVERAGE = {refresh: () => { lastKey = ""; tick(); }};
  setInterval(tick, 500);
  tick();
})();
