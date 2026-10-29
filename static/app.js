const $ = (id) => document.getElementById(id);
let examples = {};

async function init() {
  examples = await (await fetch("/api/examples")).json();
  const sel = $("example");
  Object.keys(examples).forEach((k) => sel.add(new Option(k, k)));
  sel.onchange = () => fillExample(sel.value);
  fillExample(sel.value);

  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => {
    document.querySelectorAll(".tab, .tabpane").forEach((e) => e.classList.remove("active"));
    t.classList.add("active");
    $(t.dataset.tab).classList.add("active");
  }));

  const info = await (await fetch("/api/model-info")).json();
  const m = info.metrics;
  $("perfView").innerHTML =
    `<div class="metric"><b>${m.accuracy}</b>Accuracy</div>
     <div class="metric"><b>${m.roc_auc}</b>ROC-AUC</div>
     <div class="metric"><b>${m.n_train}</b>Training variants</div>
     <div class="metric"><b>${m.n_test}</b>Test variants</div>
     <p class="msg">${m.note}</p>`;
  renderBars($("impBars"), Object.entries(info.global_importance)
    .sort((a, b) => b[1] - a[1]).map(([feature, value]) => ({ feature, value })), false);

  $("predictBtn").onclick = predict;
  predict();
}

function fillExample(name) {
  const seq = examples[name];
  $("seq").value = ">" + name.replace(/\s+/g, "_") + "\n" + (seq.match(/.{1,60}/g) || []).join("\n");
}

function renderBars(el, items, signed = true) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 0.01);
  el.innerHTML = items.map((i) => {
    const w = (Math.abs(i.value) / max) * 100;
    const cls = signed && i.value > 0 ? "fill pos" : "fill";
    const val = (signed && i.value > 0 ? "+" : "") + i.value.toFixed(2);
    return `<div class="hb"><span class="name">${i.feature}</span>
      <div class="track"><div class="${cls}" style="width:${w}%"></div></div><span>${val}</span></div>`;
  }).join("");
}

async function predict() {
  $("error").hidden = true;
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence: $("seq").value }),
  });
  const d = await res.json();
  if (!res.ok) { $("error").textContent = d.error; $("error").hidden = false; return; }
  render(d);
}

function render(d) {
  const path = d.label === "Pathogenic";
  $("result").className = "result " + (path ? "path" : "ben");
  $("cls").textContent = d.label;
  $("badge").textContent = path ? "HIGH RISK" : "LOW RISK";
  $("badge").className = "badge " + (path ? "path" : "ben");
  $("conf").textContent = d.confidence == null ? "n/a" : `${d.confidence.toFixed(2)} (${Math.round(d.confidence * 100)}%)`;
  $("confBar").style.width = (d.confidence == null ? 0 : d.confidence * 100) + "%";
  $("msg").textContent = d.message || (d.variant.extra_variants
    ? `${d.variant.extra_variants} additional variant(s) found; only the first was scored.` : "");

  if (d.shap) renderBars($("shapBars"), d.shap); else $("shapBars").innerHTML = "";

  const v = d.variant;
  const rows = [
    ["Gene", d.gene], ["Sequence Length", d.length + " bp"],
    ["Variant Position", v ? `${v.position} (nt ${v.nt_position})` : "None"],
    ["Reference Nucleotide", v ? v.ref_codon : "—"], ["Alternate Nucleotide", v ? v.alt_codon : "—"],
    ["Amino Acid Change", v ? v.aa_change : "—"],
    ["Conservation Score", v ? v.conservation.toFixed(2) + (v.conservation > 0.9 ? " (Highly conserved)" : "") : "—"],
  ];
  $("info").innerHTML = rows.map(([k, val]) => `<tr><th>${k}</th><td>${val}</td></tr>`).join("");

  renderSequence(d.sequence, v);
}

function renderSequence(seq, v) {
  const start = v ? v.codon_start : -1, end = start + 3;
  let out = "";
  for (let r = 0; r < seq.length; r += 50) {
    out += `<span class="ln">${r + 1}</span>`;
    for (let c = r; c < Math.min(r + 50, seq.length); c += 10) {
      let chunk = "";
      for (let i = c; i < Math.min(c + 10, seq.length); i++)
        chunk += i >= start && i < end ? `<mark>${seq[i]}</mark>` : seq[i];
      out += chunk + " ";
    }
    out += "\n";
  }
  $("seqBox").innerHTML = out;
}

init();
