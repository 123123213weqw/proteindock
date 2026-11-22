# -*- coding: utf-8 -*-
"""ProteinDock 命令行接口

 子命令:
   dock    执行蛋白-小分子对接
   train   训练 EGNN 深度学习打分模型
   web     启动 Web 可视化平台
   info    打印环境与版本信息
"""

import argparse
import os
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proteindock",
        description="ProteinDock 基于深度学习打分的蛋白质-小分子对接平台 v{}".format(
            __version__))
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------- dock ----------------
    p_dock = sub.add_parser("dock", help="执行蛋白-小分子对接")
    p_dock.add_argument("-r", "--receptor", required=True,
                        help="蛋白质 PDB 文件")
    p_dock.add_argument("-l", "--ligand", required=True,
                        help="配体 SDF/MOL2/PDB 文件")
    p_dock.add_argument("-o", "--out", default="proteindock_out",
                        help="输出目录(默认 proteindock_out)")
    p_dock.add_argument("--pocket", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"),
                        help="口袋中心坐标(默认自动检测)")
    p_dock.add_argument("--box", type=float, default=24.0,
                        help="搜索盒子边长 / A(默认 24)")
    p_dock.add_argument("--exhaustiveness", type=int, default=8,
                        help="独立搜索次数(默认 8)")
    p_dock.add_argument("--method", choices=["mc_sa", "ga", "hybrid"],
                        default="hybrid", help="搜索策略(默认 hybrid)")
    p_dock.add_argument("--model", default=None,
                        help="EGNN 打分模型权重路径(可选)")
    p_dock.add_argument("--dl-weight", type=float, default=1.0,
                        help="深度学习分数融合权重(默认 1.0)")
    p_dock.add_argument("--top-k", type=int, default=9,
                        help="保留前 K 个构象(默认 9)")
    p_dock.add_argument("--seed", type=int, default=2025, help="随机种子")

    # ---------------- train ----------------
    p_train = sub.add_parser("train", help="训练 EGNN 打分模型")
    p_train.add_argument("--data", required=True,
                         help="dockdata 数据集根目录")
    p_train.add_argument("--out", default="models/egnn", help="模型输出目录")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--hidden", type=int, default=96)
    p_train.add_argument("--layers", type=int, default=4)
    p_train.add_argument("--device", default=None,
                         help="cpu / cuda(默认自动)")

    # ---------------- web ----------------
    p_web = sub.add_parser("web", help="启动 Web 可视化平台")
    p_web.add_argument("--host", default="0.0.0.0")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.add_argument("--model", default=None, help="EGNN 权重路径")
    p_web.add_argument("--workdir", default="web_jobs",
                       help="任务与上传文件目录")

    # ---------------- info ----------------
    sub.add_parser("info", help="打印环境与版本信息")
    return parser


def cmd_dock(args) -> int:
    import numpy as np
    from .io.protein import parse_pdb
    from .io.ligand import load_ligand
    from .pipeline import DockingPipeline

    print("== ProteinDock v{} ==".format(__version__))
    print("受体: {}".format(args.receptor))
    print("配体: {}".format(args.ligand))
    protein = parse_pdb(args.receptor)
    ligand = load_ligand(args.ligand)
    print("蛋白重原子 {} 个 | 配体重原子 {} 个 | 可旋转键 {} 个".format(
        protein.n_atoms, ligand.n_atoms, ligand.n_torsions))

    pocket = np.array(args.pocket) if args.pocket else None
    pipeline = DockingPipeline(
        protein, ligand, pocket=pocket, box_size=args.box,
        exhaustiveness=args.exhaustiveness, method=args.method,
        model_path=args.model, dl_weight=args.dl_weight,
        device="cuda" if _cuda_available() else "cpu", seed=args.seed)

    def progress(stage, pct, msg):
        print("[{:>3}%] [{}] {}".format(pct, stage, msg))

    results = pipeline.run(progress=progress, save_dir=args.out,
                           top_k=args.top_k)
    print("\n==== 对接结果 ====")
    print("{:>4s} {:>10s} {:>10s} {:>10s} {:>8s}".format(
        "rank", "vina", "dl", "final", "cluster"))
    for res in results:
        dl = "-" if res.scores.get("dl") is None else \
            "{:.2f}".format(res.scores["dl"])
        print("{:>4d} {:>10.2f} {:>10s} {:>10.2f} {:>8d}".format(
            res.rank, res.scores["vina"], dl, res.scores["final"],
            res.cluster_size))
    print("\n结果已写入: {}/".format(args.out))
    return 0


def cmd_train(args) -> int:
    from .dl.train import train_egnn
    summary = train_egnn(args.data, args.out, epochs=args.epochs,
                         batch_size=args.batch_size, lr=args.lr,
                         hidden_dim=args.hidden, n_layers=args.layers,
                         device=args.device)
    print(json_dump(summary))
    return 0


def cmd_web(args) -> int:
    import web.app as web_app
    web_app.launch(host=args.host, port=args.port, model_path=args.model,
                   workdir=args.workdir)
    return 0


def cmd_info(args) -> int:
    import platform
    print("ProteinDock v{} on Python {} ({})".format(
        __version__, platform.python_version(), platform.system()))
    try:
        import numpy
        print("numpy      {}".format(numpy.__version__))
    except ImportError:
        print("numpy      未安装")
    try:
        import scipy
        print("scipy      {}".format(scipy.__version__))
    except ImportError:
        print("scipy      未安装")
    try:
        import rdkit
        print("rdkit      {}".format(rdkit.__version__))
    except ImportError:
        print("rdkit      未安装")
    try:
        import torch
        print("torch      {} (cuda: {})".format(
            torch.__version__, torch.cuda.is_available()))
    except ImportError:
        print("torch      未安装")
    return 0


def json_dump(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"dock": cmd_dock, "train": cmd_train,
                "web": cmd_web, "info": cmd_info}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
