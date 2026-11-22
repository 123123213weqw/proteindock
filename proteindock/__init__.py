# -*- coding: utf-8 -*-
"""ProteinDock —— 基于深度学习打分的蛋白质-小分子对接平台

 ProteinDock 集成多种前沿技术完成蛋白质-小分子柔性对接:
   1. 蒙特卡洛模拟退火(MC-SA)与遗传算法(GA)混合构象搜索
   2. Vina 式多项物理经验打分函数(高斯/静电/氢键/疏水项)
   3. 打分网格预计算加速(0.375 A 间距)
   4. E(n) 等变图神经网络(EGNN)深度学习打分与 pose 重排序
   5. Flask + WebGL(3Dmol.js) 交互式可视化平台与命令行双模式

 :author: 温州大学大学生创新创业训练计划项目组
 :license: MIT
"""

__version__ = "1.0.0"
__title__ = "ProteinDock"
__description__ = "基于深度学习打分的蛋白质-小分子对接平台"
__all__ = [
    "chemistry",
    "io",
    "scoring",
    "search",
    "dl",
    "pipeline",
    "cli",
]
