# -*- coding: utf-8 -*-
"""ProteinDock Web 可视化平台(Flask 后端)

 提供:
   - 蛋白/配体文件上传或内置示例提交对接任务
   - 后台线程池异步执行, 前端轮询进度
   - 3D 结果展示(3Dmol.js, 本地静态资源)
   - 打分明细 / 训练曲线 / 示例下载等 REST API
"""

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, render_template, request, send_from_directory

from proteindock import __version__
from proteindock.io.protein import parse_pdb
from proteindock.io.ligand import load_ligand
from proteindock.pipeline import DockingPipeline

app = Flask(__name__)

# 运行时由 launch() 注入配置
STATE = {
    "model_path": None,
    "workdir": "web_jobs",
    "device": "cpu",
    "examples_dir": None,
}
JOBS = {}                       # job_id -> dict(status/progress/message/result)
LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------- 页面
@app.route("/")
def index():
    return render_template("index.html", version=__version__)


# ---------------------------------------------------------------- API: 任务
@app.route("/api/jobs", methods=["POST"])
def create_job():
    """创建对接任务(文件上传或选择内置示例)。"""
    data = request.form
    workdir = STATE["workdir"]
    os.makedirs(os.path.join(workdir, "uploads"), exist_ok=True)
    job_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]

    rec = request.files.get("receptor")
    lig = request.files.get("ligand")
    example = data.get("example")
    if example:
        ex_dir = STATE["examples_dir"]
        if not ex_dir or not os.path.isdir(ex_dir):
            return jsonify({"error": "示例不存在"}), 404
        rec_path = os.path.join(ex_dir, example, "protein.pdb")
        lig_path = os.path.join(ex_dir, example, "ligand.sdf")
        if not (os.path.exists(rec_path) and os.path.exists(lig_path)):
            return jsonify({"error": "示例文件缺失: " + example}), 404
        job_dir = os.path.join(workdir, "jobs", job_id)
        os.makedirs(job_dir, exist_ok=True)
    else:
        if rec is None or lig is None:
            return jsonify({"error": "请同时上传蛋白 PDB 与配体 SDF"}), 400
        job_dir = os.path.join(workdir, "jobs", job_id)
        os.makedirs(job_dir, exist_ok=True)
        rec_path = os.path.join(job_dir, "input_receptor.pdb")
        lig_path = os.path.join(job_dir, "input_ligand.sdf")
        rec.save(rec_path)
        lig.save(lig_path)
        if os.path.getsize(rec_path) > 30 * 1024 * 1024:
            return jsonify({"error": "蛋白文件过大(>30MB)"}), 400

    params = {
        "pocket": None,
        "box_size": float(data.get("box_size", 24)),
        "exhaustiveness": int(data.get("exhaustiveness", 8)),
        "method": data.get("method", "hybrid"),
        "dl_weight": float(data.get("dl_weight", 1.0)),
        "seed": int(data.get("seed", 2025)),
    }
    if data.get("pocket"):
        try:
            params["pocket"] = [float(v) for v in
                                data.get("pocket").split(",")]
        except ValueError:
            return jsonify({"error": "口袋坐标格式应为 x,y,z"}), 400

    with LOCK:
        JOBS[job_id] = {
            "status": "queued", "progress": 0, "stage": "queue",
            "message": "排队中", "params": params, "dir": job_dir,
            "created": time.time(),
        }
    POOL.submit(_run_docking_job, job_id, rec_path, lig_path, params)
    return jsonify({"job_id": job_id})


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    with LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "任务不存在"}), 404
    payload = {k: job.get(k) for k in
               ("status", "progress", "stage", "message", "error", "result")}
    return jsonify(payload)


@app.route("/api/jobs/<job_id>/files/<path:name>")
def job_file(job_id, name):
    with LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "任务不存在"}), 404
    # 注意: Flask 的 send_from_directory 会把相对目录解析为
    # 相对 app.root_path(即 web/ 模块目录), 必须转绝对路径
    out_dir = os.path.abspath(os.path.join(job["dir"], "output"))
    if not os.path.isfile(os.path.join(out_dir, name)):
        return jsonify({"error": "文件不存在"}), 404
    return send_from_directory(out_dir, name, as_attachment=True)


# ---------------------------------------------------------------- API: 元信息
@app.route("/api/examples")
def list_examples():
    ex_dir = STATE["examples_dir"]
    items = []
    if ex_dir and os.path.isdir(ex_dir):
        for name in sorted(os.listdir(ex_dir)):
            meta = os.path.join(ex_dir, name, "meta.json")
            info = {"id": name}
            if os.path.exists(meta):
                with open(meta, encoding="utf-8") as fh:
                    info.update(json.load(fh))
            items.append(info)
    return jsonify({"examples": items})


@app.route("/api/model")
def model_info():
    """EGNN 模型元信息与训练曲线。"""
    model_path = STATE["model_path"]
    if not model_path or not os.path.exists(model_path):
        return jsonify({"loaded": False})
    history_path = os.path.join(os.path.dirname(model_path), "history.json")
    history = None
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as fh:
            history = json.load(fh)
    return jsonify({"loaded": True, "path": model_path,
                    "history": history})


@app.route("/api/version")
def version():
    return jsonify({"name": "ProteinDock", "version": __version__,
                    "device": STATE["device"]})


# ---------------------------------------------------------------- 任务执行
def _run_docking_job(job_id, rec_path, lig_path, params):
    """线程池工作函数: 执行完整对接并写结果。"""
    with LOCK:
        JOBS[job_id].update(status="running", stage="prepare",
                            progress=3, message="解析输入结构...")

    def progress(stage, pct, msg):
        with LOCK:
            JOBS[job_id].update(stage=stage, progress=pct, message=msg)

    try:
        protein = parse_pdb(rec_path)
        ligand = load_ligand(lig_path)
        pipeline = DockingPipeline(
            protein, ligand, pocket=params["pocket"],
            box_size=params["box_size"],
            exhaustiveness=params["exhaustiveness"],
            method=params["method"],
            model_path=STATE["model_path"],
            dl_weight=params["dl_weight"],
            device=STATE["device"], seed=params["seed"])
        out_dir = os.path.join(JOBS[job_id]["dir"], "output")
        results = pipeline.run(progress=progress, save_dir=out_dir)

        # 结果 JSON(含各 pose 的 SDF 路径与打分明细)
        result = {
            "n_atoms_protein": protein.n_atoms,
            "n_atoms_ligand": ligand.n_atoms,
            "n_torsions": ligand.n_torsions,
            "stats": pipeline.stats,
            "poses": [
                {
                    "rank": r.rank,
                    "scores": r.scores,
                    "cluster_size": r.cluster_size,
                    "sdf": "pose_{}.sdf".format(r.rank),
                } for r in results
            ],
            "complex_pdb": "complex_best.pdb",
        }
        with LOCK:
            JOBS[job_id].update(status="done", progress=100, stage="done",
                                message="对接完成", result=result)
    except Exception as exc:              # noqa: BLE001 - 全量回传给前端
        with LOCK:
            JOBS[job_id].update(status="error", stage="error",
                                message=str(exc), error=str(exc))


# ---------------------------------------------------------------- 启动
def _pick_device() -> str:
    """实际执行一次 GPU 矩阵乘法验证 CUDA 可用性。

    部分环境下 torch.cuda.is_available() 为 True 但驱动/架构不匹配
    (如 CUDA 13 包与 Volta 架构), 真正计算时才报错; 试算可提前发现。
    """
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.randn(16, 16, device="cuda")
            (a @ a).sum().item()
            return "cuda"
    except Exception:
        pass
    return "cpu"


def launch(host="0.0.0.0", port=8000, model_path=None, workdir="web_jobs",
           examples_dir=None):
    """CLI 入口: 启动 Web 服务。"""
    STATE["model_path"] = model_path
    STATE["workdir"] = workdir
    STATE["device"] = _pick_device()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    STATE["examples_dir"] = examples_dir or os.path.join(root, "examples")
    os.makedirs(os.path.join(workdir, "jobs"), exist_ok=True)
    print("ProteinDock Web v{} | device={} | model={}".format(
        __version__, STATE["device"], model_path or "(纯经验打分)"))
    print("示例目录: {}".format(STATE["examples_dir"]))
    print("访问: http://{}:{}/  (Ctrl+C 退出)".format(
        "127.0.0.1" if host == "0.0.0.0" else host, port))
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    import sys
    launch(port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
