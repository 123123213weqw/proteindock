# ProteinDock 基于深度学习打分的蛋白质-小分子对接平台

![version](https://img.shields.io/badge/version-1.0-blue) ![python](https://img.shields.io/badge/python-3.8%2B-green)

ProteinDock 是面向药物设计场景的蛋白质-小分子分子对接计算平台,
由温州大学大学生创新创业训练计划项目组开发。

## 技术特点

| 模块 | 技术 |
|------|------|
| 打分函数 | Vina 式五项物理经验打分(高斯吸引/斥力/疏水/氢键) |
| 深度学习 | E(n) 等变图神经网络(EGNN) pose 打分, 知识蒸馏训练 |
| 构象搜索 | 蒙特卡洛模拟退火(MC-SA) + 遗传算法(GA) 混合策略 |
| 加速 | 0.375 A 打分网格预计算 + 三线性插值 |
| 可视化 | Flask + WebGL(3Dmol.js) 交互式 3D 平台 |
| 接口 | Web 平台与命令行(CLI)双模式 |

## 安装

```bash
python3 -m venv .venv --system-site-packages   # 复用系统 torch(可选)
source .venv/bin/activate
pip install -r requirements.txt
```

## 命令行用法

```bash
# 对接(自动检测口袋)
python -m proteindock.cli dock -r examples/1m17/protein.pdb \
      -l examples/1m17/ligand.sdf -o out_1m17 --exhaustiveness 8

# 构建 EGNN 训练数据集并训练
python scripts/build_dataset.py --out datasets/dockdata --n-decoys 24
python -m proteindock.cli train --data datasets/dockdata --out models/egnn

# 启动 Web 平台
python -m proteindock.cli web --model models/egnn/best.pt --port 8000
```

## Web 平台

浏览器访问 `http://<server>:8600/`, 选择内置示例或上传
蛋白 PDB 与配体 SDF, 设置搜索参数后提交任务, 实时查看进度,
对接完成后可交互查看 3D 构象(cartoon/stick/surface)、
打分排名与能量分解。

## 目录结构

```
proteindock/
├── proteindock/        # 核心 Python 包
│   ├── chemistry/      # 元素/原子分型/刚体几何
│   ├── io/             # PDB 与 SDF 解析
│   ├── scoring/        # 经验打分/网格/混合打分
│   ├── search/         # MC-SA 与 GA 搜索引擎
│   ├── dl/             # EGNN 模型/数据集/训练
│   ├── pipeline.py     # 对接流水线
│   └── cli.py          # 命令行入口
├── web/                # Flask Web 平台
├── scripts/            # 数据集构建 / 软著材料生成
├── tests/              # 单元测试
└── examples/           # 示例体系
```

## License

MIT
