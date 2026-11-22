# -*- coding: utf-8 -*-
"""软件著作权申请材料生成

 1. 源代码文档: 按中国版权保护中心要求, 提交前 30 页 + 后 30 页
    (每页 50 行), 自动汇总项目源码并排版为 txt;
 2. 软件登记信息表骨架(名称/版本/开发环境等自动填充)。

 用法: python scripts/make_docs.py --out docs/软著材料
"""

import argparse
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 源代码收集顺序(核心引擎在前, 前端在后)
SOURCE_FILES = [
    "proteindock/__init__.py",
    "proteindock/chemistry/elements.py",
    "proteindock/chemistry/atom_types.py",
    "proteindock/chemistry/geometry.py",
    "proteindock/io/protein.py",
    "proteindock/io/ligand.py",
    "proteindock/scoring/vina_sf.py",
    "proteindock/scoring/grid.py",
    "proteindock/scoring/hybrid.py",
    "proteindock/search/mc_sa.py",
    "proteindock/search/genetic.py",
    "proteindock/dl/features.py",
    "proteindock/dl/model.py",
    "proteindock/dl/dataset.py",
    "proteindock/dl/train.py",
    "proteindock/pipeline.py",
    "proteindock/cli.py",
    "web/app.py",
    "web/static/app.js",
    "web/static/style.css",
    "web/templates/index.html",
    "scripts/build_dataset.py",
    "scripts/make_docs.py",
]

LINES_PER_PAGE = 50
PAGES_EACH_SIDE = 30
HEADER = """
{title}
源 程 序

软件名称: {name}
版本号: V1.0
共 {total_lines} 行 / {total_pages} 页
本文件包含前 {front_pages} 页与后 {back_pages} 页源程序
生成日期: {today}
===============================================================
"""


def collect_lines():
    """按顺序收集全部源代码行(带文件分隔注释)。"""
    lines = []
    for rel in SOURCE_FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print("警告: 缺少 {}".format(rel))
            continue
        lines.append("/* ====== 文件: {} ====== */".format(rel))
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                text = raw.rstrip("\n")
                # 屏蔽个人路径信息
                text = text.replace("/Users/", "/**/")
                lines.append(text if text.strip() else "")
        lines.append("")
    return lines


def make_source_doc(out_dir, title):
    lines = collect_lines()
    total_pages = (len(lines) + LINES_PER_PAGE - 1) // LINES_PER_PAGE
    os.makedirs(out_dir, exist_ok=True)
    header = HEADER.format(
        title=title, name=title, total_lines=len(lines),
        total_pages=total_pages, front_pages=PAGES_EACH_SIDE,
        back_pages=PAGES_EACH_SIDE, today=date.today().isoformat())

    front = lines[:PAGES_EACH_SIDE * LINES_PER_PAGE]
    back = lines[-PAGES_EACH_SIDE * LINES_PER_PAGE:] if total_pages > 60 else []
    doc_lines = header.splitlines() + [""] + front
    if back:
        doc_lines += ["", "......(中间部分略)......", ""] + back

    path = os.path.join(out_dir, "源代码_前30页后30页.txt")
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(0, len(doc_lines), LINES_PER_PAGE):
            page_no = i // LINES_PER_PAGE + 1
            page = doc_lines[i:i + LINES_PER_PAGE]
            fh.write("\n".join(page) + "\n")
            fh.write("[第 {} 页]\n\n".format(page_no))
    print("源代码文档: {} (总行数 {} => {} 页)".format(
        path, len(lines), total_pages))
    return path, len(lines), total_pages


def make_info_sheet(out_dir, title, n_lines, n_pages):
    """软件登记基本信息表骨架。"""
    info = """# 软件著作权登记信息(草稿)

软件全称: {title}
软件简称: ProteinDock
版本号: V1.0
开发完成日期: {today}
首次发表日期: 未发表
开发方式: 独立开发(大学生创新创业训练计划项目成果)
软件用途与技术特点:
  本软件面向药物设计场景, 提供蛋白质-小分子分子对接计算平台。
  集成 Vina 式五项物理经验打分函数、E(n) 等变图神经网络(EGNN)
  深度学习打分模型、蒙特卡洛模拟退火与遗传算法混合构象搜索、
  打分网格加速与 WebGL 三维交互可视化, 支持 Web 与命令行双模式。

源程序量: {n_lines} 行(约 {n_pages} 页, 每页 50 行)
开发环境:
  操作系统: Ubuntu 22.04 (NVIDIA Tesla V100 GPU 服务器)
  编程语言: Python 3.10 / JavaScript (ES6)
  核心依赖: NumPy, SciPy, RDKit, PyTorch, Flask, 3Dmol.js
运行环境:
  操作系统: Linux / macOS / Windows
  Python 3.8+, 支持 CUDA 的 NVIDIA GPU(可选, CPU 亦可运行)
""".format(title=title, today=date.today().isoformat(),
           n_lines=n_lines, n_pages=n_pages)
    path = os.path.join(out_dir, "软件登记信息表_草稿.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(info)
    print("登记信息表: {}".format(path))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "软著材料"))
    ap.add_argument("--title",
                    default="基于深度学习打分的蛋白质-小分子对接平台软件")
    args = ap.parse_args()
    _, n_lines, n_pages = make_source_doc(args.out, args.title)
    make_info_sheet(args.out, args.title, n_lines, n_pages)


if __name__ == "__main__":
    main()
