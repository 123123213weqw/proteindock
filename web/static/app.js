/* ProteinDock Web 前端逻辑: 任务提交 / 轮询 / 3D 渲染 / 图表 */
"use strict";

const $ = (sel) => document.querySelector(sel);

/* ---------------- 全局状态 ---------------- */
const App = {
  jobId: null,
  pollTimer: null,
  poses: [],
  selectedPose: 0,
  receptorPdb: "",      // 原始受体文本(渲染用)
  viewer: null,
  viewMode: "cartoon",
};

/* ---------------- 初始化 ---------------- */
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initDropzones();
  loadExamples();
  loadModelInfo();
  loadVersion();
  $("#btn-submit").addEventListener("click", submitJob);
  $("#view-cartoon").onclick = () => setViewMode("cartoon");
  $("#view-stick").onclick = () => setViewMode("stick");
  $("#view-surface").onclick = () => setViewMode("surface");
  $("#view-reset").onclick = () => { if (App.viewer) App.viewer.zoomTo(); };
});

function loadVersion() {
  fetch("/api/version").then(r => r.json()).then(d => {
    $("#version-badge").textContent = "v" + d.version;
    $("#device-badge").textContent = d.device.toUpperCase() === "CUDA" ? "GPU 加速" : "CPU";
    $("#device-badge").classList.toggle("gpu", d.device === "cuda");
  }).catch(() => {});
}

/* ---------------- Tab 切换 ---------------- */
function initTabs() {
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      $("#tab-example").classList.toggle("hidden", btn.dataset.tab !== "example");
      $("#tab-upload").classList.toggle("hidden", btn.dataset.tab !== "upload");
    });
  });
}

/* ---------------- 文件上传区 ---------------- */
function initDropzones() {
  bindDrop("drop-receptor", "file-receptor", "name-receptor");
  bindDrop("drop-ligand", "file-ligand", "name-ligand");
}
function bindDrop(zoneId, inputId, nameId) {
  const zone = $("#" + zoneId), input = $("#" + inputId);
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files.length) $("#" + nameId).textContent = input.files[0].name;
  });
  ["dragover", "dragenter"].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove("dragover"); }));
  zone.addEventListener("drop", e => {
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      $("#" + nameId).textContent = e.dataTransfer.files[0].name;
    }
  });
}

/* ---------------- 示例与模型信息 ---------------- */
function loadExamples() {
  fetch("/api/examples").then(r => r.json()).then(d => {
    const sel = $("#example-select");
    sel.innerHTML = "";
    App.examples = d.examples;
    d.examples.forEach(ex => {
      const opt = document.createElement("option");
      opt.value = ex.id;
      opt.textContent = ex.title || ex.id;
      if (ex.description) opt.title = ex.description;
      sel.appendChild(opt);
    });
    if (!d.examples.length) {
      const opt = document.createElement("option");
      opt.textContent = "(无内置示例)";
      sel.appendChild(opt);
    } else {
      sel.selectedIndex = 0;        // listbox 模式不会自动选中第一项
      fillExamplePocket(d.examples[0]);
      sel.addEventListener("change", () => {
        const ex = (App.examples || []).find(e => e.id === sel.value);
        fillExamplePocket(ex);
      });
    }
  }).catch(() => {});
}

/* 选中示例时自动填入推荐口袋坐标(来自晶体配体质心) */
function fillExamplePocket(ex) {
  if (ex && ex.pocket) {
    $("#param-pocket").value = ex.pocket.map(v => v.toFixed(1)).join(",");
  }
}

function loadModelInfo() {
  fetch("/api/model").then(r => r.json()).then(d => {
    if (!d.loaded) return;
    const s = d.history && d.history.summary;
    $("#model-summary").innerHTML =
      "EGNN 等变图神经网络已加载 (epochs " + (s ? s.epochs : "?") +
      ", val loss " + (s ? s.best_val_loss.toFixed(4) : "?") +
      ", 训练体系 " + (s ? s.n_systems : "?") + " 个)";
    if (d.history && d.history.history) drawLossCurve(d.history.history);
  }).catch(() => {});
}

/* ---------------- 任务提交与轮询 ---------------- */
function submitJob() {
  const useExample = !$("#tab-example").classList.contains("hidden");
  const fd = new FormData();
  if (useExample) {
    const ex = $("#example-select").value;
    if (!ex) return alert("请选择示例体系");
    fd.append("example", ex);
  } else {
    const r = $("#file-receptor").files[0], l = $("#file-ligand").files[0];
    if (!r || !l) return alert("请上传蛋白 PDB 与配体文件");
    fd.append("receptor", r); fd.append("ligand", l);
  }
  fd.append("method", $("#param-method").value);
  fd.append("exhaustiveness", $("#param-exhaustiveness").value);
  fd.append("box_size", $("#param-box").value);
  fd.append("dl_weight", $("#param-dlweight").value);
  const pocket = $("#param-pocket").value.trim();
  if (pocket) fd.append("pocket", pocket);

  $("#btn-submit").disabled = true;
  $("#progress-wrap").classList.add("active");
  setProgress(0, "queue", "提交任务...");
  fetch("/api/jobs", { method: "POST", body: fd })
    .then(r => r.json())
    .then(d => {
      if (d.error) throw new Error(d.error);
      App.jobId = d.job_id;
      pollStatus();
    })
    .catch(e => {
      setProgress(0, "error", e.message);
      $("#btn-submit").disabled = false;
    });
}

function pollStatus() {
  App.pollTimer = setInterval(() => {
    fetch("/api/jobs/" + App.jobId).then(r => r.json()).then(job => {
      setProgress(job.progress || 0, job.stage, job.message);
      if (job.status === "done") {
        clearInterval(App.pollTimer);
        onJobDone(job);
      } else if (job.status === "error") {
        clearInterval(App.pollTimer);
        setProgress(100, "error", "失败: " + job.error);
      }
    }).catch(() => {});
  }, 1500);
}

/* 阶段名中文映射 */
const STAGE_LABELS = {
  queue: "排队中", prepare: "解析输入", grid: "构建打分网格",
  search: "构象搜索", cluster: "聚类去重", rescore: "深度学习重打分",
  done: "已完成", error: "失败",
};

function setProgress(pct, stage, msg) {
  $("#progress-fill").style.width = pct + "%";
  $("#progress-pct").textContent = pct + "%";
  $("#progress-stage").textContent = STAGE_LABELS[stage] || stage;
  $("#progress-msg").textContent = msg || "";
}

/* ---------------- 结果展示 ---------------- */
function onJobDone(job) {
  $("#btn-submit").disabled = false;
  App.poses = job.result.poses;
  App.selectedPose = 0;

  // 结果表
  const tbody = $("#result-table tbody");
  tbody.innerHTML = "";
  job.result.poses.forEach(p => {
    const tr = document.createElement("tr");
    tr.className = "data-row" + (p.rank === 1 ? " selected" : "");
    tr.dataset.rank = p.rank;
    const dl = p.scores.dl === null ? "-" : p.scores.dl.toFixed(2);
    tr.innerHTML = "<td>" + p.rank + "</td><td>" + p.scores.vina.toFixed(2) +
      "</td><td>" + dl + "</td><td><b>" + p.scores.final.toFixed(2) +
      "</b></td><td>" + p.cluster_size + "</td>";
    tr.addEventListener("click", () => selectPose(p.rank));
    tbody.appendChild(tr);
  });

  // 统计信息
  const st = job.result.stats;
  $("#stats-box").textContent =
    "蛋白重原子: " + job.result.n_atoms_protein +
    "\n配体重原子: " + job.result.n_atoms_ligand +
    "\n可旋转键: " + job.result.n_torsions +
    "\n口袋中心: " + st.pocket_center.map(v => v.toFixed(1)).join(", ") +
    "\n搜索策略: " + st.method + " × " + st.n_runs + " 次独立搜索" +
    "\n聚类代表构象: " + st.n_cluster_representatives +
    "\nDL 模型: " + (st.dl_model_loaded ? "已加载" : "未加载") +
    "\n总耗时: " + st.elapsed_seconds + " s";

  // 3D 展示最优 pose
  renderResult(job.result);
  drawEnergy(job.result.poses[0]);
}

async function selectPose(rank) {
  App.selectedPose = rank;
  document.querySelectorAll(".result-table tr.data-row").forEach(tr => {
    tr.classList.toggle("selected", Number(tr.dataset.rank) === rank);
  });
  const pose = App.poses.find(p => p.rank === rank);
  if (!pose) return;
  const resp = await fetch("/api/jobs/" + App.jobId + "/files/" + pose.sdf);
  const sdfText = await resp.text();
  renderLigandPose(sdfText);
  drawEnergy(pose);
}

/* ---------------- 3D 渲染 ---------------- */
function initViewer() {
  const el = $("#viewer");
  el.innerHTML = "";
  App.viewer = $3Dmol.createViewer(el, { backgroundColor: "#0a0e15" });
  return App.viewer;
}

async function renderResult(result) {
  const v = initViewer();
  const pdbResp = await fetch("/api/jobs/" + App.jobId + "/files/complex_best.pdb");
  const pdbText = await pdbResp.text();
  App.receptorPdb = pdbText.replace(/HETATM.*LIG.*\n/g, "");
  applyViewMode();
  const sdfResp = await fetch("/api/jobs/" + App.jobId + "/files/pose_1.sdf");
  renderLigandPose(await sdfResp.text());
}

function renderLigandPose(sdfText) {
  const v = App.viewer;
  if (v.$ligand) v.removeModel(v.$ligand);
  const lig = v.addModel(sdfText, "sdf");
  v.$ligand = lig;
  lig.setStyle({}, { stick: { colorscheme: "goldCarbon", radius: 0.16 } });
  v.render();
}

function applyViewMode() {
  const v = App.viewer;
  if (!v || !App.receptorPdb) return;
  if (v.$receptor) v.removeModel(v.$receptor);
  const rec = v.addModel(App.receptorPdb, "pdb");
  v.$receptor = rec;
  if (App.viewMode === "cartoon") {
    rec.setStyle({}, { cartoon: { color: "spectrum" } });
  } else if (App.viewMode === "stick") {
    rec.setStyle({}, { stick: { colorscheme: "cyanCarbon", radius: 0.08 },
                       line: { color: "#3d4a66" } });
  } else if (App.viewMode === "surface") {
    rec.setStyle({}, {});
    v.addSurface($3Dmol.SurfaceType.VDW, {
      opacity: 0.85, color: "#4f8cff"
    }, { model: rec });
  }
  v.zoomTo({ model: rec });
  v.render();
}

function setViewMode(mode) {
  App.viewMode = mode;
  document.querySelectorAll(".viewer-controls .chip").forEach(c =>
    c.classList.remove("active"));
  $("#view-" + mode).classList.add("active");
  if (App.viewer) {
    App.viewer.removeAllSurfaces();
    applyViewMode();
    if (App.viewer.$ligand) renderLigandPoseLast();
  }
}

function renderLigandPoseLast() {
  /* 视角切换后保持配体可见(最近一次 pose 已在 viewer 中) */
  App.viewer.render();
}

/* ---------------- 图表 ---------------- */
function drawEnergy(pose) {
  // 注意: 服务端 report.json 里有五项分解, 前端先用总分近似展示
  const canvas = $("#energy-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("#energy-pose-label").textContent = "pose #" + pose.rank;
  const items = [
    ["经验总分", pose.scores.vina, "#4f8cff"],
    ["相互作用", pose.scores.vina_inter, "#22c1a4"],
    ["配体内能", pose.scores.vina_intra, "#f5a623"],
  ];
  if (pose.scores.dl !== null && pose.scores.dl !== undefined)
    items.push(["EGNN 分", pose.scores.dl, "#c084fc"]);
  drawBars(ctx, items, canvas);
}

function drawBars(ctx, items, canvas) {
  const w = canvas.width, h = canvas.height, pad = 34;
  const vmin = Math.min(0, ...items.map(i => i[1]));
  const vmax = Math.max(0.1, ...items.map(i => i[1]));
  const y0 = h - pad;
  const scale = (h - pad * 2) / (vmax - vmin);
  ctx.strokeStyle = "#3a4560";
  ctx.beginPath(); ctx.moveTo(pad, y0 - (0 - vmin) * scale);
  ctx.lineTo(w - 8, y0 - (0 - vmin) * scale); ctx.stroke();
  const bw = (w - pad * 2) / items.length * 0.55;
  items.forEach((it, i) => {
    const cx = pad + (w - pad * 2) * (i + 0.5) / items.length;
    const val = it[1];
    const yTop = y0 - (Math.max(val, 0) - vmin) * scale;
    const yBot = y0 - (Math.min(val, 0) - vmin) * scale;
    ctx.fillStyle = it[2];
    ctx.fillRect(cx - bw / 2, yTop, bw, Math.max(yBot - yTop, 2));
    ctx.fillStyle = "#8b93a7";
    ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(it[0], cx, h - 12);
    ctx.fillText(val.toFixed(2), cx, yTop - 6);
  });
}

function drawLossCurve(history) {
  const canvas = $("#loss-canvas");
  canvas.classList.remove("hidden");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, pad = 30;
  ctx.clearRect(0, 0, w, h);
  const all = history.train.concat(history.val);
  const vmax = Math.max(...all), vmin = Math.min(...all);
  const x = (i, n) => pad + (w - pad * 1.4) * i / Math.max(n - 1, 1);
  const y = (v) => h - pad - (h - pad * 1.8) * (v - vmin) / Math.max(vmax - vmin, 1e-6);
  [["train", "#4f8cff"], ["val", "#22c1a4"]].forEach(([key, color]) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
    history[key].forEach((v, i) => {
      const px = x(i, history[key].length), py = y(v);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
  });
  ctx.fillStyle = "#8b93a7"; ctx.font = "11px sans-serif";
  ctx.fillText("train (蓝) / val (绿) loss", pad, 16);
}
