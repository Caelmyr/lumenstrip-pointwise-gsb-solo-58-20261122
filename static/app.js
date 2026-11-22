"use strict";

const root = document.querySelector("#content");
const dialog = document.querySelector("#dialog");
const dialogBody = document.querySelector("#dialog-body");
const status = document.querySelector("#notice");
const loc = document.querySelector("#location");
const models = {
  station: null, sources: [], records: [], tools: [], active: null, selected: [],
  tab: "inspect", tool: "spectrum", fields: {}, report: null, asset: "", witness: null,
  reportSource: 0, strip: null, calibration: null, savedFits: [], bookmarks: [],
  prepOverlay: "", draft: null, expected: "", pick: null, lastPoint: null, activity: [], folio: [],
};
let timeout;

function esc(v) {
  return String(v ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}
const q = (s) => document.querySelector(s);
const title = (s) => String(s).replace(/[_-]/g, " ");
const pretty = (v) => JSON.stringify(v, null, 2);
const display = (v) => v == null ? "—" : typeof v === "boolean" ? v ? "Yes" : "No"
  : typeof v === "object" ? JSON.stringify(v) : String(v);
const source = () => models.sources.find((s) => s.id === models.active);
const tool = () => models.tools.find((t) => t.id === models.tool);
const studioPath = () => "/api/stations/" + models.station.id;
const sourcePath = () => "/api/sources/" + models.active;

async function rpc(url, options = {}) {
  const config = {...options};
  if (options.json !== undefined) {
    config.headers = {"Content-Type":"application/json"};
    config.body = JSON.stringify(options.json);
    delete config.json;
  }
  const response = await fetch(url, config);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw Error(data.error || "Request failed (" + response.status + ").");
  return data;
}
const post = (url, json) => rpc(url, {method:"POST", json});

function announce(message, bad=false) {
  status.hidden = false;
  status.className = bad ? "alert" : "";
  status.textContent = message;
  window.clearTimeout(timeout);
  if (!bad) timeout = setTimeout(() => { status.hidden = true; }, 2800);
}
function safe(work) { return Promise.resolve().then(work).catch((e) => announce(e.message, true)); }

function button(label, action, key="", className="") {
  return `<button type="button" class="${className}" data-act="${action}" data-key="${esc(key)}">${esc(label)}</button>`;
}
function openDialog(name, body) {
  dialogBody.innerHTML = `<h2>${esc(name)}</h2>${body}`;
  dialog.showModal();
}
function field(label, name, value="", opts="") {
  return `<label>${esc(label)}<input name="${esc(name)}" value="${esc(value)}" ${opts}></label>`;
}
function nfield(label,name,value,step="any") { return field(label,name,value,`type="number" step="${step}"`); }
function area(label, name, value, rows=3) {
  return `<label class="wide">${esc(label)}<textarea name="${esc(name)}" rows="${rows}">${esc(value)}</textarea></label>`;
}
function checked(label, key, value) {
  return `<label class="check"><input type="checkbox" name="${esc(key)}" ${value ? "checked" : ""}>${esc(label)}</label>`;
}
function dropdown(label, key, options, current, extra="") {
  return `<label>${esc(label)}<select name="${esc(key)}" ${extra}>${options.map(([value, text]) =>
    `<option value="${esc(value)}" ${current === value ? "selected" : ""}>${esc(text)}</option>`).join("")}</select></label>`;
}
function form(name, inner, submit, disabled=false) {
  return `<form data-form="${name}">${inner}<div class="buttons"><button class="primary" ${disabled ? "disabled" : ""}>${esc(submit)}</button></div></form>`;
}
function folds(label, value) {
  return `<details><summary>${esc(label)}</summary><pre>${esc(pretty(value))}</pre></details>`;
}
function metrics(values) {
  return `<div class="metrics">${Object.entries(values || {}).filter(([,v]) => v == null || typeof v !== "object")
    .slice(0, 18).map(([k,v]) => `<div class="metric"><small>${esc(title(k))}</small><strong>${esc(display(v))}</strong></div>`).join("")}</div>`;
}
function table(rows) {
  if (!rows?.length) return '<p class="tiny">No recorded rows.</p>';
  const names = [...new Set(rows.slice(0, 50).flatMap(Object.keys))].slice(0, 18);
  return '<div class="table-wrap"><table><thead><tr>' + names.map((k) => `<th>${esc(title(k))}</th>`).join("") +
    "</tr></thead><tbody>" + rows.slice(0, 160).map((r) => {
    const point = r.source_uv || r.start_uv || r.source_1_uv;
    const value = r.position ?? r.x_axis ?? r.crest_axis;
    const tag = point ? ` class="clickable" data-act="row-point" data-key="${esc(JSON.stringify(point))}"`
      : typeof value === "number" ? ` class="clickable" data-act="row-axis" data-key="${value}"` : "";
    return `<tr${tag}>${names.map((k) => `<td>${esc(display(r[k]))}</td>`).join("")}</tr>`;
  }).join("") + "</tbody></table></div>";
}

async function home() {
  models.station = null;
  models.report = null;
  loc.textContent = "";
  const {stations} = await rpc("/api/stations");
  root.innerHTML = `<section class="home"><h1>Read light from the original.</h1><p>Trace camera spectra, check wavelength support, compare a filter with its open reference, and retain actual camera evidence.</p>
  <div class="row">${button("New studio","new-studio","","primary")} ${button("Open synthetic demo","demo")}</div>
  ${stations.length ? `<div class="station-grid">${stations.map((x) => `<button class="station-card" data-act="station" data-key="${x.id}">
    <strong>${esc(x.name)}</strong><span>${x.images} camera images · ${x.reports} readings${x.synthetic ? " · Synthetic" : ""}</span>
    <span>${esc(x.notes)}</span></button>`).join("")}</div>` : ""}</section>`;
}
function cleanReport() {
  models.report = null;
  models.witness = null;
  models.asset = "";
  models.reportSource = 0;
}
async function loadStation(key, keep=false) {
  const data = await rpc("/api/stations/" + key);
  Object.assign(models, {station:data.station,sources:data.sources,records:data.records});
  if (!keep) {
    models.active = data.sources[0]?.id || null;
    models.selected = models.active ? [models.active] : [];
    models.tab = "inspect"; models.tool = "spectrum"; models.fields = {}; cleanReport();
  }
  if (!models.active && data.sources.length) models.active = data.sources[0].id;
  if (models.active) await loadSource();
  loc.textContent = models.station.name;
  render();
}
async function loadSource() {
  if (!models.active) return;
  const output = await rpc(sourcePath());
  models.sources = models.sources.map((s) => s.id === output.source.id ? output.source : s);
  models.savedFits = output.calibrations;
  models.bookmarks = output.bookmarks;
  models.strip = structuredClone(output.source.strip);
  const currentFit = output.calibrations.find((x) => x.id === output.source.active_fit);
  models.calibration = currentFit ? structuredClone(currentFit.calibration) : {title:"Wavelength ruler", method:"linear", limit:"anchors",
    anchors:[{name:"First line", column:.2, nm:450, weight:1},{name:"Second line", column:.8, nm:650, weight:1}]};
  models.expected = ""; models.pick = null; models.prepOverlay = ""; models.lastPoint = null;
}
function sourceList() {
  const list = models.sources.map((s) => `<div class="source-card ${s.id === models.active ? "current" : ""}">
    <button type="button" data-act="source" data-key="${s.id}"><img src="${esc(s.files.preview)}" alt=""></button>
    <div><button type="button" data-act="source" data-key="${s.id}"><strong>${esc(s.meta.title)}</strong><small>${esc(s.meta.role)} · ${s.meta.exposure_ms} ms${s.meta.synthetic ? " · Synthetic" : ""}</small></button>
    <label class="check"><input data-select-source="${s.id}" type="checkbox" aria-label="Use ${esc(s.meta.title)}"
      ${models.selected.includes(s.id) ? "checked" : ""}>Use ${models.selected.includes(s.id) ? models.selected.indexOf(s.id)+1 : ""}</label></div></div>`).join("");
  return `<aside class="panel"><h2>Camera images</h2>${button("Import image","upload","","primary")} ${button("Generate synthetic","synthetic")}${list}</aside>`;
}
function assetList() {
  if (!models.report) return [];
  const indexed = Object.fromEntries((models.report.asset_index || []).map((a) => [a.name,a.title]));
  return (models.report.assets || []).map((a) => ({...a,title:a.title || indexed[a.name] || a.name}));
}
function frame() {
  const chosen = assetList().find((a) => a.name === models.asset);
  const viewing = models.tab === "inspect" || models.tab === "records";
  const img = viewing && chosen ? chosen.url : models.prepOverlay || source()?.files.preview;
  const figureTitle = viewing && chosen ? chosen.title : source()?.meta.title || "Camera stage";
  const showSource = viewing && models.report?.sources?.length > 1;
  const images = assetList().filter((a) => a.mime.startsWith("image/"));
  const label = models.pick ? "Click the requested position on the camera image." : "";
  return `<section class="stage"><h2>${esc(figureTitle)}</h2>${showSource ? dropdown("Point evidence source","report-source",
    models.report.sources.map((s,i) => [String(i),s.meta.title]),String(models.reportSource)) : ""}
    ${img ? `<div class="scene ${models.pick ? "is-picking" : ""}"><div class="target">
    <img id="stage-image" alt="${esc(figureTitle)}" src="${esc(img)}"><svg class="draft-layer" id="draft-layer" aria-hidden="true"></svg>
    </div></div>` : '<p class="tiny">Import a camera image to begin.</p>'}
    ${label ? `<p class="tiny">${label}</p>` : ""}
    ${viewing && images.length ? `<div class="assets">${images.map((a) => button(a.title,"asset",a.name,a.name === models.asset ? "on" : "")).join("")}</div>` : ""}
    ${witnessView()}</section>`;
}
function witnessView() {
  const w = models.witness;
  if (!w) return "";
  const result = w.reading;
  return `<section class="measure"><h3>Camera witness</h3>${metrics({state:result.state, requested_axis:result.requested_axis,
    sampled_axis:result.sampled_axis, source:result.source?.meta.title, column:result.pixel?.[0], row:result.pixel?.[1]})}
    ${w.witness ? `<img class="inline-witness" alt="Magnified actual source pixels" src="${esc(w.witness)}">` : ""}
        ${result.bookmark ? `<p class="tiny">${esc(result.bookmark.name)} · ${esc(result.bookmark.message)}</p>` : ""}
    ${result.uv && models.tab === "strip" ? button("Bookmark this source point","bookmark") : ""}
    ${folds("Exact native evidence", result.column_result || result)}</section>`;
}
function opt(spec, value) {
  const {key, title:label, kind} = spec;
  if (kind === "number") return nfield(label + (spec.units ? " (" + spec.units + ")" : ""),key,value,spec.step);
  if (kind === "boolean") return checked(label,key,value);
  if (kind === "choice") return dropdown(label,key,spec.choices.map((v) => [v,title(v)]),value);
  const more = kind === "crop" ? button("Mark two corners","tool-draw",key) : "";
  return `<div class="wide">${area(label,key,pretty(value),kind === "crop" ? 3 : 4)}${more}</div>`;
}
function inspect() {
  if (!source() || !tool()) return '<aside class="panel right"><h2>Inspect</h2><p class="tiny">Choose a camera source.</p></aside>';
  const current = tool();
  const groups = [...new Set(models.tools.map((t) => t.section))];
  const selectors = groups.map((g) => `<optgroup label="${esc(g)}">${models.tools.filter((t) => t.section === g)
    .map((t) => `<option value="${t.id}" ${current.id === t.id ? "selected" : ""}>${esc(t.label)}</option>`).join("")}</optgroup>`).join("");
  const editing = '<div class="fields">' + current.settings.map((spec) => opt(spec,models.fields[spec.key] ?? spec.default)).join("") + "</div>";
  return `<aside class="panel right"><h2>Inspect</h2><label>Reading tool<select id="tool-selector">${selectors}</select></label>
    <p class="help">${esc(current.description)}</p>${current.sources[0] > 1 ? `<p class="tiny">Choose ${current.sources.join("–")} images using the checkboxes. The index is the input order.</p>` : ""}
    ${form("inspect",editing,"Run reading")}${resultView()}</aside>`;
}
function resultView() {
  const r = models.report;
  if (!r) return "";
  const assets = assetList();
  const files = assets.filter((a) => a.mime.includes("csv")).map((a) => `<a class="download" download="${esc(a.name)}" href="${esc(a.url)}">${esc(a.title)}</a>`).join("");
  const controls = r.id ? `<a class="download" target="_blank" href="/api/records/${r.id}/sheet">Print sheet</a>
    <a class="download" href="/api/records/${r.id}/download">Evidence ZIP</a><a class="download" href="/api/records/${r.id}/download?originals=0">Measurements only</a>
    ${button("Add review note","review")} ${button("Check saved sources","check-record")}` : button("Retain reading","retain","","primary");
  return `<section class="measure"><h3>${esc(r.name || r.title)}</h3><p class="tiny">${esc(r.summary)}</p>${metrics(r.metrics)}
  <div class="buttons measure">${controls}${files}</div><details><summary>Measured rows · ${r.row_count}</summary>${table(r.rows)}</details>
  ${folds("Inputs and source versions",r.sources)}${folds("All measurements",r.metrics)}
  ${(r.warnings||[]).map((w) => `<p class="warning">${esc(w)}</p>`).join("")}
  ${(r.review_notes||[]).map((w) => `<p class="notes-entry">${esc(w.message)}</p>`).join("")}</section>`;
}
function stripEditor() {
  if (!source()) return "";
  const a = models.strip;
  const checks = checked("Subtract measured sidebands","subtract_sides",a.subtract_sides);
  const editor = `<div class="fields">${area("Center anchors (relative x, y)","endpoints",pretty(a.endpoints),3)}
    ${nfield("Half-height (px)","half_height",a.half_height,1)}${nfield("Column step (px)","step",a.step,1)}
    ${nfield("Sideband gap (px)","side_spacing",a.side_spacing,1)}${nfield("Sideband height (px)","side_height",a.side_height,1)}
    ${nfield("Shadow guard","low_guard",a.low_guard,.0001)}${nfield("Clip guard","high_guard",a.high_guard,.001)}
    ${dropdown("Signal channel","channel",["linear-luma","red","green","blue","mean"].map((x)=>[x,title(x)]),a.channel)}
    ${dropdown("Estimator","level",[["mean","Mean"],["median","Median"]],a.level)}${checks}</div>`;
  const notes = models.bookmarks.map((m) => button(m.title,"bookmark-open",m.id,"record-select")).join("");
  return `<aside class="panel right"><h2>Strip geometry</h2><p class="help">Set two points in the middle of the spectrum, then inspect the sideband rows.</p>
    ${button("Click two center anchors","strip-draw")}${form("strip",editor,"Preview strip")}${button("Save strip","strip-retain")}
    ${button("Edit capture","edit-capture")} ${button("Check camera original","check-camera")}${notes ? `<h3>Source bookmarks</h3>${notes}` : ""}</aside>`;
}
function calibrationEditor() {
  if (!source()) return "";
  const p = models.calibration;
  const f = `<div class="fields">${field("Calibration name","title",p.title)}${dropdown("Fit","method",[["linear","Linear least squares"],["interpolate","Interpolate marked lines"]],p.method)}
    ${dropdown("Wavelength support","limit",[["anchors","Within anchor span"],["frame","Entire image width"]],p.limit)}</div>`;
  const anchors = p.anchors.map((a,i) => `<div class="anchor-item">${field("Name","anchor-name-"+i,a.name)}${nfield("Wavelength (nm)","anchor-nm-"+i,a.nm)}
    ${nfield("Relative column","anchor-column-"+i,a.column,.000001)}${button("Pick","anchor-pick",i)}</div>`).join("");
  const choices = models.savedFits.map((x) => `<div class="fit-entry">${button(x.calibration.title,"fit-use",x.id,source().active_fit === x.id ? "primary" : "")}</div>`).join("");
  return `<aside class="panel right"><h2>Wavelength ruler</h2>
    ${form("calibration",f+anchors+button("Add anchor","anchor-add")+button("Remove last anchor","anchor-remove"),"Preview ruler")}
    ${button("Save ruler","calibration-retain")}${button("Import ruler CSV","fit-import")}${button("Use native columns","fit-use","")}
    ${source().active_fit ? `<a class="download" href="${sourcePath()}/calibration.csv">Download ruler CSV</a>` : ""}
    ${choices ? `<h3>Saved rulers</h3>${choices}` : ""}${models.draft?.fit ? folds("Fit residuals",models.draft.fit) : ""}</aside>`;
}
function records() {
  const folio = models.records.map((r) => checked("Compare " + r.name, "compare-" + r.id,
    models.folio.includes(r.id)).replace('type="checkbox"', 'type="checkbox" data-folio="' + r.id + '"')).join("");
  const comparator = models.records.length > 1 ? '<details><summary>Compare saved readings</summary>' + folio +
    button("Open printable comparison","folio","","download") + '</details>' : "";
  return `<aside class="panel right"><h2>Retained readings</h2>${models.records.length ? models.records.map((r) => button(r.name,"record",r.id,"record-select")).join("") : '<p class="tiny">No readings retained.</p>'}${comparator}${resultView()}</aside>`;
}
function activity() {
  return `<aside class="panel right"><h2>Activity</h2>${models.activity.map((e) => `<p class="notes-entry">${esc(e.type)}<br><small>${esc(e.created)}</small></p>`).join("")}</aside>`;
}
function render() {
  const st = models.station;
  if (!st) return;
  const tabs = [["inspect","Inspect"],["strip","Strip"],["calibration","Wavelength"],["records","Retained"],["activity","Activity"]];
  root.innerHTML = `<header class="title"><div><h1>${esc(st.name)}</h1><p class="tiny">${esc(st.notes)}${st.synthetic ? " · Synthetic images" : ""}</p></div>${button("Edit studio","edit-studio")}</header>
    <div class="tabs" role="tablist">${tabs.map(([i,label]) => `<button role="tab" aria-selected="${i===models.tab}" data-act="tab" data-key="${i}">${label}</button>`).join("")}</div>
    <div class="workspace">${sourceList()}${frame()}${({inspect,strip:stripEditor,calibration:calibrationEditor,records,activity})[models.tab]()}</div>`;
  const image = q("#stage-image");
  if (image) {
    image.addEventListener("load", svgPoints);
    if (image.complete) svgPoints();
  }
}

const sourceRoles = ["sample","open-reference","dark-reference","calibration","repeat"];
function captureDialog(edit=false) {
  const meta = edit ? source().meta : {title:"",role:"sample",exposure_ms:30,notes:"",synthetic:false};
  openDialog(edit ? "Edit camera metadata" : "Import camera image", form(edit ? "edit-capture" : "upload",
    (!edit ? '<label>Choose PNG, JPEG, TIFF or WebP<input type="file" name="image" accept=".png,.jpeg,.jpg,.tif,.tiff,.webp" required></label>' : "") +
    field("Observation name","title",meta.title,"required maxlength=90") +
    dropdown("Capture role","role",sourceRoles.map((id)=>[id,title(id)]),meta.role) +
    nfield("Exposure (ms)","exposure_ms",meta.exposure_ms) + area("Notes","notes",meta.notes) +
    checked("Synthetic image","synthetic",meta.synthetic), edit ? "Save metadata" : "Import"));
}
function stationDialog(edit=false) {
  const st = edit ? models.station : {name:"",notes:""};
  openDialog(edit ? "Edit studio" : "New studio",form(edit ? "edit-studio" : "new-studio",
    field("Studio name","name",st.name,"required maxlength=85") + area("Studio notes","notes",st.notes),
    edit ? "Save studio" : "Create studio"));
}
function readCapture(fd) {
  return {title:fd.get("title"), role:fd.get("role"), exposure_ms:Number(fd.get("exposure_ms")),
    notes:fd.get("notes"), synthetic:fd.has("synthetic")};
}
function rememberSettings() {
  const editor = q('form[data-form="inspect"]');
  if (!editor || !tool()) return;
  const fd = new FormData(editor);
  const cfg = {};
  for (const spec of tool().settings) {
    const value = fd.get(spec.key);
    cfg[spec.key] = spec.kind === "number" ? Number(value) : spec.kind === "boolean" ? fd.has(spec.key)
      : spec.kind === "choice" ? value : JSON.parse(value);
  }
  models.fields = cfg;
}
function readStrip() {
  const fd = new FormData(q('form[data-form="strip"]'));
  const data = {endpoints:JSON.parse(fd.get("endpoints")), channel:fd.get("channel"), level:fd.get("level"),
    subtract_sides:fd.has("subtract_sides")};
  for (const k of ["half_height","step","side_spacing","side_height","low_guard","high_guard"]) data[k]=Number(fd.get(k));
  models.strip = data;
  return data;
}
function readCalibration() {
  const fd = new FormData(q('form[data-form="calibration"]'));
  const data = {title:fd.get("title"), method:fd.get("method"), limit:fd.get("limit"),
    anchors:models.calibration.anchors.map((old,i)=>({name:fd.get("anchor-name-"+i),
      column:Number(fd.get("anchor-column-"+i)),nm:Number(fd.get("anchor-nm-"+i)),weight:old.weight || 1}))};
  models.calibration = data;
  return data;
}
async function stripPreview(retain) {
  const settings = readStrip();
  const preview = await post(sourcePath()+"/strip",{revision:source().revision,strip:settings,retain});
  if (retain) {
    await loadSource();
    announce("Strip saved.");
  }
  models.prepOverlay = preview.overlay;
  models.pick = null; models.witness = null;
  render();
}
async function fitPreview(retain) {
  const settings = readCalibration();
  const output = await post(sourcePath()+"/calibration",{revision:source().revision, calibration:settings,
    retain, expected:models.expected});
  if (retain) {
    await loadSource();
    announce("Wavelength ruler saved.");
  } else models.expected = output.preview_key;
  models.draft = output;
  models.prepOverlay = output.overlay;
  models.pick = null; models.witness = null;
  render();
}
function applyReport(result) {
  models.report = result;
  models.asset = assetList().find((item)=>item.mime.startsWith("image/"))?.name || "";
  models.witness = null; models.reportSource = 0;
}
async function runInspection(form) {
  rememberSettings();
  const t = tool();
  const sources = t.sources[1] === 1 ? [source().id] : models.selected;
  if (!sources || sources.length < t.sources[0] || sources.length > t.sources[1])
    throw Error("Choose " + t.sources.join("–") + " camera images.");
  const submit = form.querySelector('button[type="submit"],button.primary');
  submit.disabled = true;
  try {
    const result = await post(studioPath()+"/tools/"+t.id,{sources,settings:models.fields});
    applyReport(result);
    render();
    announce("Reading ready.");
  } finally {submit.disabled = false;}
}
async function retainReading(name) {
  const r = models.report;
  const saved = await post(studioPath()+"/tools/"+r.tool,{...r.order,name,retain:true,expected:r.fingerprint});
  applyReport(saved);
  await loadStation(models.station.id,true);
  models.tab = "records";
  render();
  announce("Reading retained.");
}
async function showRecord(key) {
  const output = await rpc("/api/records/"+key);
  applyReport(output);models.tab="records";render();
}
async function pointWitness(space, value, forcedIndex) {
  const report = ["inspect","records"].includes(models.tab) ? models.report : null;
  const index = forcedIndex ?? models.reportSource;
  const id = report?.sources?.[index]?.id ?? models.active;
  if (!id) return;
  const url = report?.id ? "/api/records/"+report.id+"/probe" : "/api/sources/"+id+"/probe";
  const doc = {on:space,value,source_index:index};
  if (report?.sources?.[index]?.revision) doc.expected_revision = report.sources[index].revision;
  const response = await post(url,doc);
  models.lastPoint = space === "camera" ? value : response.reading.uv;
  models.witness = response;
  const priorScroll = window.scrollY;
  render();
  window.scrollTo(0,priorScroll);
}
function svgPoints() {
  const image = q("#stage-image");
  const svg = q("#draft-layer");
  if (!image || !svg) return;
  const box = image.getBoundingClientRect();
  svg.setAttribute("width",box.width);
  svg.setAttribute("height",box.height);
  svg.setAttribute("viewBox",`0 0 ${box.width} ${box.height}`);
  const points = models.pick?.points ?? [];
  svg.innerHTML = points.map(([x,y])=>`<circle cx="${x*box.width}" cy="${y*box.height}" r="5" fill="#8fe3cf" stroke="#14252b" stroke-width="2"/>`).join("");
}
async function stagePoint(event) {
  const img = q("#stage-image");
  const box = img.getBoundingClientRect();
  const uv = [Math.min(1,Math.max(0,(event.clientX-box.left)/box.width)),
    Math.min(1,Math.max(0,(event.clientY-box.top)/box.height))];
  if (models.pick) {
    const current = models.pick;
    current.points.push(uv.map((v)=>Math.round(v*1e6)/1e6));
    if (current.kind === "anchor") {
      models.calibration.anchors[current.index].column=current.points[0][0];
      models.expected="";
    } else if (current.points.length < 2) {
      svgPoints();return;
    } else if (current.kind === "strip") models.strip.endpoints=current.points;
    else models.fields[current.key]=current.points;
    models.pick=null;
    render();
    return;
  }
  const r = models.report;
  const probe = ["inspect","records"].includes(models.tab) && r ? r.probes?.[models.asset] : null;
  if (probe?.kind === "axis") {
    const [left,top,right,bottom] = probe.plot_uv;
    if (uv[0] < left || uv[0] > right || uv[1] < top || uv[1] > bottom) return;
    return pointWitness("axis",probe.x[0]+(uv[0]-left)/(right-left)*(probe.x[1]-probe.x[0]));
  }
  if (probe?.kind === "source-crop") {
    const [a,b,c,d] = probe.bounds;
    const meta = r.sources[probe.source_index].image;
    return pointWitness("camera",[(a+uv[0]*(c-a))/meta.width,(b+uv[1]*(d-b))/meta.height],probe.source_index);
  }
  if (probe?.kind === "source") return pointWitness("camera",uv,probe.source_index);
  if (!r || !models.asset || !["inspect","records"].includes(models.tab)) return pointWitness("camera",uv);
}
function markDraw(kind,key="") {
  if (kind === "strip") readStrip();
  if (kind === "anchor") readCalibration();
  if (kind === "tool") {
    rememberSettings();
    cleanReport();
  }
  models.prepOverlay="";
  models.pick={kind,key,index:Number(key),points:[]};
  render();
}

const actions = {
  home,
  "new-studio":()=>stationDialog(false),
  "edit-studio":()=>stationDialog(true),
  station:loadStation,
  demo:async()=>{announce("Opening demonstration images…");const r=await post("/api/demo",{});await loadStation(r.station.id);},
  packet:()=>openDialog("Check evidence ZIP",form("packet",'<label>Choose a local evidence packet<input type="file" accept=".zip" name="packet" required></label>',"Check packet")+'<div id="packet-result"></div>'),
  upload:()=>captureDialog(false),
  synthetic:()=>openDialog("Synthetic detector image",form("synthetic",
    field("Observation name","title","Synthetic calibration strip","required")+
    '<div class="grid2">'+nfield("Exposure (ms)","exposure",30)+nfield("Strip width parameter (px)","sigma",3)+'</div>'+
    area("Line descriptions","lines",pretty([{name:"Green line",column:.5,height:.7,width:3,color:"green"},
      {name:"Blue line",column:.24,height:.5,width:2,color:"blue"},{name:"Red line",column:.73,height:.42,width:4,color:"red"}]),8)+
    checked("Import this generated image","retain",false),"Generate image")+'<div id="synthetic-preview"></div>'),
  "edit-capture":()=>captureDialog(true),
  "check-camera":async()=>{
    const {health}=await rpc(sourcePath()+"/health");
    openDialog("Camera integrity",metrics(health)+folds("Original image evidence",health)+
      `<p><a class="download" href="${sourcePath()}/original">Download original</a></p>`);
  },
  source:async(key)=>{
    if (models.tab==="inspect" && tool()) rememberSettings();
    models.active=key;if(tool()?.sources[1]===1)models.selected=[key];
    cleanReport();await loadSource();render();
  },
  tab:async(key)=>{
    if (models.tab==="inspect" && tool()) rememberSettings();
    if(key==="activity") models.activity=(await rpc(studioPath()+"/activity")).activity;
    models.tab=key;models.pick=null;models.prepOverlay="";models.witness=null;render();
  },
  asset:(key)=>{models.asset=key;models.witness=null;render();},
  record:showRecord,
  retain:()=>openDialog("Retain reading",form("retain",field("Reading name","name",models.report.title,"required maxlength=88"),"Save reading")),
  review:()=>openDialog("Add review note",form("review",area("Review note","message","",4),"Save note")),
  "check-record":async()=>{
    const result = await rpc("/api/records/" + models.report.id + "/health?originals=1");
    openDialog("Saved source check", metrics(result.health) +
      folds("Actual native observations",result.health.sources) +
      folds("Saved measurement files",result.health.assets));
  },
  folio:()=>{
    if(models.folio.length<2 || models.folio.length>4)throw Error("Choose two to four retained readings.");
    const search=new URLSearchParams();
    for(const item of models.folio)search.append("reading",item);
    window.open(studioPath()+"/folio?"+search.toString(),"_blank","noopener");
  },
  "row-point":(key)=>pointWitness("camera",JSON.parse(key)),
  "row-axis":(key)=>pointWitness("axis",Number(key)),
  "strip-draw":()=>markDraw("strip"),
  "tool-draw":(key)=>markDraw("tool",key),
  "anchor-pick":(key)=>markDraw("anchor",key),
  "strip-retain":()=>stripPreview(true),
  "calibration-retain":()=>fitPreview(true),
  "fit-import":()=>openDialog("Import ruler anchors",form("fit-import",
    '<label>Choose a UTF-8 anchor CSV<input type="file" name="anchors" accept=".csv" required></label>'+field("Calibration name","title","Imported wavelength table")+
    dropdown("Fit","method",[["linear","Linear least squares"],["interpolate","Interpolate marked lines"]],"linear")+
    dropdown("Wavelength support","limit",[["anchors","Within anchor span"],["frame","Entire image width"]],"anchors"),"Preview import")),
  "fit-use":async(key)=>{await post(sourcePath()+"/calibrations/select",{revision:source().revision,fit_id:key||null});
    cleanReport();await loadSource();render();announce("Active ruler updated.");},
  "anchor-add":()=>{const c=readCalibration();if(c.anchors.length<28)c.anchors.push({name:"Line "+(c.anchors.length+1),column:.55,nm:550,weight:1});
    models.expected="";render();},
  "anchor-remove":()=>{const c=readCalibration();if(c.anchors.length>2)c.anchors.pop();models.expected="";render();},
  bookmark:()=>openDialog("Bookmark source point",form("bookmark",field("Bookmark name","name","","required")+
    area("Reading notes","message",""),"Save bookmark")),
  "bookmark-open":async(key)=>{
    const replay=await rpc(sourcePath()+"/bookmarks/"+key);
    models.witness=replay;models.lastPoint=replay.reading.uv;render();
  },
};

const forms = {
  "new-studio":async(fd)=>{const result=await post("/api/stations",{name:fd.get("name"),notes:fd.get("notes")});
    dialog.close();await loadStation(result.station.id);},
  "edit-studio":async(fd)=>{await rpc(studioPath(),{method:"PATCH",json:{name:fd.get("name"),notes:fd.get("notes"),revision:models.station.revision}});
    dialog.close();await loadStation(models.station.id,true);},
  upload:async(fd)=>{
    const data=new FormData();
    data.set("image",fd.get("image"));data.set("capture",JSON.stringify(readCapture(fd)));
    const result=await rpc(studioPath()+"/sources",{method:"POST",body:data});
    models.active=result.source.id;models.selected=[models.active];cleanReport();dialog.close();
    await loadStation(models.station.id,true);announce("Camera image imported.");
  },
  synthetic:async(fd)=>{
    const r=await post(studioPath()+"/synthetic",{retain:fd.has("retain"),
      pattern:{title:fd.get("title"),exposure:Number(fd.get("exposure")),sigma:Number(fd.get("sigma")),lines:JSON.parse(fd.get("lines"))}});
    if(r.source) {
      models.active=r.source.id; models.selected=[models.active]; cleanReport();dialog.close();
      await loadStation(models.station.id,true);announce("Synthetic image imported.");
    } else {
      q("#synthetic-preview").innerHTML='<img class="inline-witness" alt="Generated detector preview" src="'+esc(r.preview)+'">';
    }
  },
  "edit-capture":async(fd)=>{await post(sourcePath()+"/capture",{revision:source().revision,capture:readCapture(fd)});
    dialog.close();await loadSource();render();announce("Capture saved.");},
  inspect:(fd,form)=>runInspection(form),
  strip:()=>stripPreview(false),
  calibration:()=>fitPreview(false),
  retain:async(fd)=>{dialog.close();await retainReading(fd.get("name"));},
  review:async(fd)=>{const key=models.report.id;await post("/api/records/"+key+"/review",{message:fd.get("message")});
    dialog.close();await showRecord(key);},
  "fit-import":async(fd)=>{
    fd.set("metadata",JSON.stringify({title:fd.get("title"),fit:fd.get("method"),limit:fd.get("limit")}));
    const r=await rpc(sourcePath()+"/calibration/import",{method:"POST",body:fd});
    models.calibration=r.settings; models.draft=r;models.expected=r.preview_key;models.prepOverlay=r.overlay;
    dialog.close();render();announce("Imported anchors are ready to inspect.");
  },
  bookmark:async(fd)=>{await post(sourcePath()+"/bookmarks",{uv:models.lastPoint,name:fd.get("name"),message:fd.get("message"),revision:source().revision});
    dialog.close();await loadSource();render();announce("Bookmark saved.");},
  packet:async(fd)=>{const result=await rpc("/api/packet-audit",{method:"POST",body:fd});
    q("#packet-result").innerHTML=`<h3>${result.audit.valid ? "Packet checks passed" : "Packet disagrees with its index"}</h3>`+
      metrics({reading:result.audit.reading,members:result.audit.member_count})+table(result.audit.members);},
};

document.addEventListener("click",(event)=> {
  const target=event.target.closest("[data-act]");
  if(target && actions[target.dataset.act]) safe(()=>actions[target.dataset.act](target.dataset.key));
  else if(event.target.id==="stage-image")safe(()=>stagePoint(event));
});
document.addEventListener("submit",(event)=> {
  const f=event.target;
  if(!f.dataset.form || !forms[f.dataset.form])return;
  event.preventDefault();
  safe(()=>forms[f.dataset.form](new FormData(f),f));
});
document.addEventListener("change",(event)=> {
  const node=event.target;
  if(node.id==="tool-selector") {
    models.tool=node.value;models.fields={};cleanReport();render();
  } else if(node.dataset.selectSource) {
    if(tool() && models.tab==="inspect") {
      try { rememberSettings(); } catch(error) { announce(error.message,true); return; }
    }
    const key=node.dataset.selectSource;
    models.selected=node.checked ? [...models.selected,key] : models.selected.filter((id)=>id!==key);
    cleanReport();render();
  } else if(node.dataset.folio) {
    const id=node.dataset.folio;
    models.folio=node.checked ? [...models.folio,id] : models.folio.filter((x)=>x!==id);
  } else if(node.name==="report-source") {
    models.reportSource=Number(node.value);models.witness=null;render();
  }
});
safe(async()=>{models.tools=(await rpc("/api/tools")).tools;await home();});
