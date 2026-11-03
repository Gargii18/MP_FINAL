const DATA = (typeof BIO !== "undefined") ? BIO : require("./data.js");
const ORDER = ["LC", "NT", "VU", "EN", "CR", "EX"];
const LABEL = {LC: "Least Concern", NT: "Near Threatened", VU: "Vulnerable", EN: "Endangered", CR: "Critically Endangered", EX: "Extinct", EW: "Extinct in the Wild", DD: "Data Deficient"};
const COLOR = {LC: "#4c9f50", NT: "#a6c34a", VU: "#f2c230", EN: "#f28c28", CR: "#d64545", EX: "#4a4a4a", EW: "#6b4a8a", DD: "#8a8a8a"};
const TIER = {LC: "Low", NT: "Low", VU: "Medium", EN: "High", CR: "High", EX: "High"};
const GEN_REASONS = ["Habitat loss and degradation (agriculture, logging, urban growth)", "Hunting, poaching and illegal wildlife trade", "Overfishing and fishing bycatch", "Invasive species and disease", "Pollution and chemical contamination", "Climate change and extreme weather", "Small, fragmented populations with low genetic diversity"];
const GEN_MEASURES = ["Protect and restore critical habitat and create wildlife corridors", "Enforce anti-poaching and wildlife-trade laws", "Use wildlife-safe fishing gear and set catch limits", "Control invasive species and manage disease", "Reduce pollution and pesticide use", "Cut emissions and protect climate-resilient habitat", "Monitor populations; use captive breeding and reintroduction where numbers are very small", "Work with local communities through incentives and human-wildlife conflict programmes"];

function summarise(table, status, known) {
  const rows = Object.entries(table).sort((a, b) => b[1] - a[1]);
  const r0 = ORDER.indexOf(status);
  const sum = f => r0 < 0 ? null : rows.filter(([c]) => f(ORDER.indexOf(c))).reduce((a, [, v]) => a + v, 0);
  return {top: rows[0][0], conf: rows[0][1], rows, known,
          worse: sum(i => i > r0), better: sum(i => i < r0), same: sum(i => i === r0)};
}

function predict(group, status) {
  const known = Object.prototype.hasOwnProperty.call(DATA.model, group);
  const table = (known ? DATA.model[group] : DATA.model.__unseen__)[status];
  if (!table) throw new Error("Invalid status: " + status);
  return summarise(table, status, known);
}

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const pc = x => (x * 100).toFixed(1) + "%";
  const badge = s => `<span class="badge" style="background:${COLOR[s]}">${s}</span> ${LABEL[s]}`;


  if (DATA.demo) $("demo").innerHTML = "⚠️ <b>Demo data:</b> predictions are illustrative only and are not real findings.";
  const byName = new Map();
  const idx = DATA.species.map(s => {
    if (s[1] && !byName.has(s[1].toLowerCase())) byName.set(s[1].toLowerCase(), s);
    if (!byName.has(s[0].toLowerCase())) byName.set(s[0].toLowerCase(), s);
    return ((s[1] || "") + "|" + s[0]).toLowerCase();
  });
  $("name").addEventListener("input", () => {
    const t = $("name").value.trim().toLowerCase();
    if (t.length < 2) { $("names").innerHTML = ""; return; }
    const out = [];
    for (let i = 0; i < idx.length && out.length < 8; i++) if (idx[i].includes(t)) out.push(DATA.species[i]);
    $("names").innerHTML = out.map(s => `<option value="${esc(s[1] || s[0])}">${esc(s[0])}</option>`).join("");
  });
  $("status").innerHTML = ORDER.map(s => `<option value="${s}">${s} – ${LABEL[s]}</option>`).join("");
  $("group").innerHTML = DATA.groups.map(g => `<option>${esc(g)}</option>`).join("") + '<option value="__unseen__">Other / not in training data</option>';

  $("form").addEventListener("submit", e => {
    e.preventDefault();
    const q = $("name").value.trim().toLowerCase();
    const sp = byName.get(q);
    let status, group, title, plain, extra;
    if (sp) {
      [status, group, plain] = [sp[3], sp[2], sp[1] ? `${sp[1]} (${sp[0]})` : sp[0]];
      title = sp[1] ? `${esc(sp[1])} (<i>${esc(sp[0])}</i>)` : `<i>${esc(sp[0])}</i>`;
      extra = `<p class="meta">Current status: ${badge(status)} · Group: ${esc(group)}${sp[4] ? ` · Recorded next status: <b>${sp[4]}</b>` : ""}</p>`;
    } else {
      if (!$("manual").open) {
        $("manual").open = true;
        $("result").innerHTML = `<p class="err">“${esc($("name").value)}” is not in the species list. Choose its current status and taxonomic group above, then press Predict again.</p>`;
        return;
      }
      [status, group, plain] = [$("status").value, $("group").value, $("name").value.trim()];
      title = esc(plain);
      extra = `<p class="meta">Using your inputs: ${badge(status)} · Group: ${esc(group === "__unseen__" ? "Other" : group)}</p>`;
    }
    if (status === "EX") {
      $("result").innerHTML = `<h2>${title}</h2>${extra}<p>This species is already <b>Extinct</b>; no further prediction applies.</p>`;
      return;
    }
    if (!(sp && sp[8]) && !["LC", "NT", "VU", "EN", "CR"].includes(status)) {
      $("result").innerHTML = `<h2>${title}</h2>${extra}<p>The IUCN status of this species is <b>${esc(LABEL[status] || status)}</b>, so no next-category prediction is available.</p>`;
      return;
    }
    const r = (sp && sp[8]) ? summarise(sp[8].p, "", true) : predict(group, status);
    const bars = r.rows.map(([c, v]) => `<div class="bar"><span>${c}</span><div class="track"><div class="fill" style="width:${v * 100}%;background:${COLOR[c]}"></div></div><span>${pc(v)}</span></div>`).join("");
    const si = sp && sp[7];
    const inf = si && (si.t || (si.h || []).length) ? `<p class="meta"><b>Population trend:</b> ${esc(si.t || "Unknown")}${si.h && si.h.length ? ` · <b>Habitats:</b> ${esc(si.h.join("; "))}` : ""}${si.y ? ` <span class="note">(IUCN assessment ${esc(si.y)})</span>` : ""}</p>` : "";
    let risk = "", report = `Species: ${plain}\nCurrent status: ${status} (${LABEL[status]})\nPredicted next status: ${r.top} (${LABEL[r.top]})\nConfidence: ${pc(r.conf)}\n`;
    if (["EN", "CR"].includes(status) || ["EN", "CR"].includes(r.top)) {
      const own = sp && sp[5] && sp[5].length;
      const reasons = own ? sp[5] : GEN_REASONS, measures = sp && sp[6] && sp[6].length ? sp[6] : GEN_MEASURES;
      const li = a => a.map(t => `<li>${esc(t)}</li>`).join("");
      risk = `<div class="reasons"><h3>Reasons for extinction risk${own ? "" : " (general)"}</h3><ul>${li(reasons)}</ul></div>` +
             `<div class="measures"><h3>Measures to conserve it${sp && sp[6] && sp[6].length ? "" : " (general)"}</h3><ul>${li(measures)}</ul></div>`;
      report += `\nReasons for extinction risk:\n- ${reasons.join("\n- ")}\n\nConservation measures:\n- ${measures.join("\n- ")}\n`;
    }
    $("result").innerHTML = `<h2>${title}</h2>${extra}
      <p>${sp && sp[8] ? "Estimated IUCN status from species traits:" : "Predicted next IUCN status:"}</p><p>${badge(r.top)}</p>
      <p>Confidence ${pc(r.conf)} · Risk tier: <b>${TIER[r.top]}</b>${r.worse === null ? "" : ` · Worsen ${pc(r.worse)} / Improve ${pc(r.better)} / Same ${pc(r.same)}`}</p>
      ${bars}${inf}${risk}${r.known ? "" : '<p class="note">⚠️ Group not in training data; prediction relies mainly on current status.</p>'}
      <p class="note">${sp && sp[8] ? "Estimated from the species’ traits (body size, geographic range, climate, human pressure, diet, life history); scores are adjusted so the many Least Concern species do not hide rarer, higher-risk categories; this is not a forecast of change. Threats, habitats and population trend come from the IUCN assessment and are not model inputs." : "Probabilities assume the status changes at the next assessment. Reasons and measures are informational and do not affect the prediction."}</p>`;
  });
}
if (typeof module !== "undefined") module.exports = {predict};
