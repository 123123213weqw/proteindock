# -*- coding: utf-8 -*-
"""对接主流程(Pipeline)

 把输入准备、口袋检测、打分网格、混合构象搜索、聚类去重、
 深度学习重打分与结果输出串联为完整流水线:

   PDB/SDF 输入
        |
   蛋白/配体解析与分型 ---> 口袋检测(可选自动)
        |                        |
   打分网格构建 <----------------+
        |
   混合构象搜索 (exhaustiveness x 独立运行:
        MC-SA 与 GA 交替执行)
        |
   RMSD 聚类去重 -> Top-K pose
        |
   EGNN 深度学习重打分 -> 加权融合排序
        |
   结果输出 (SDF / report.json / 轨迹)
"""

import json
import os
import time

import numpy as np

from .io.protein import Protein, parse_pdb, detect_pocket
from .io.ligand import Ligand, load_ligand, write_sdf
from .scoring.grid import ScoreGrid, DEFAULT_BOX_SIZE
from .scoring.hybrid import HybridScorer
from .search.mc_sa import Pose, simulated_annealing
from .search.genetic import GeneticSearch
from .chemistry.geometry import rmsd


class DockingResult:
    """单个输出构象。"""

    def __init__(self, rank, coords, scores, cluster_size=1):
        self.rank = rank
        self.coords = coords
        self.scores = scores          # dict: vina/vina_inter/vina_intra/dl/final
        self.cluster_size = cluster_size

    def to_dict(self):
        return {
            "rank": self.rank,
            "cluster_size": self.cluster_size,
            "scores": self.scores,
        }


class DockingPipeline:
    """蛋白质-小分子对接流水线。

    :param protein: Protein 对象
    :param ligand: Ligand 对象
    :param pocket: 口袋中心坐标; None 时自动检测
    :param box_size: 搜索盒子边长 / A
    :param exhaustiveness: 独立搜索次数
    :param method: mc_sa / ga / hybrid(默认, GA+SA 交替)
    :param model_path: EGNN 权重路径(可空, 纯经验打分)
    :param dl_weight: DL 融合权重
    :param device: torch 设备
    :param seed: 随机种子
    """

    def __init__(self, protein, ligand, pocket=None,
                 box_size=DEFAULT_BOX_SIZE, exhaustiveness=8,
                 method="hybrid", model_path=None, dl_weight=1.0,
                 device="cpu", seed=2025):
        self.protein = protein
        self.ligand = ligand
        self.box_size = float(box_size)
        self.exhaustiveness = int(exhaustiveness)
        self.method = method
        self.dl_weight = float(dl_weight)
        self.device = device
        self.seed = seed

        self.pocket = (np.asarray(pocket, dtype=np.float64)
                       if pocket is not None else detect_pocket(protein))

        self.grid = ScoreGrid(self.pocket, size=self.box_size)
        self.scorer = self._build_scorer(model_path)
        self.results = []
        self.stats = {}

    # ------------------------------------------------------------ 构建
    def _build_scorer(self, model_path):
        dl_model = None
        if model_path and os.path.exists(model_path):
            import torch
            from .dl.model import EGNNPoseScorer
            dl_model = EGNNPoseScorer.load_pretrained(model_path,
                                                      device=self.device)
            if dl_model is None:
                print("警告: 模型权重加载失败, 使用纯经验打分")
        return HybridScorer(self.grid, dl_model=dl_model,
                            protein=self.protein, center=self.pocket,
                            dl_weight=self.dl_weight, device=self.device)

    # ------------------------------------------------------------ 执行
    def run(self, progress=None, save_dir=None, top_k=9):
        """执行完整对接流程。

        :param progress: 可选回调 func(stage, pct, message)
        :param save_dir: 结果输出目录(None 不落盘)
        :param top_k: 保留前 K 个构象
        :return: list[DockingResult]
        """
        t0 = time.time()

        def report(stage, pct, msg):
            if progress is not None:
                progress(stage, pct, msg)

        report("grid", 5, "构建打分网格(0.375 A)...")
        self.grid.build(self.protein.coords, self.protein.type_ids)

        report("search", 15, "混合构象搜索({} 次独立运行)..."
               .format(self.exhaustiveness))
        poses = self._search(report)

        report("cluster", 80, "RMSD 聚类去重...")
        clustered = self._cluster(poses, cutoff=2.0)

        report("rescore", 88, "深度学习重打分...")
        results = []
        for rank, (pose, size) in enumerate(clustered[:top_k], start=1):
            coords = pose.coords(self.ligand)
            scores = self.scorer.score(self.ligand, coords)
            results.append(DockingResult(rank, coords, scores, size))
        self.results = results

        self.stats = {
            "elapsed_seconds": round(time.time() - t0, 2),
            "pocket_center": [round(float(v), 3) for v in self.pocket],
            "box_size": self.box_size,
            "n_runs": self.exhaustiveness,
            "n_cluster_representatives": len(clustered),
            "n_torsions": self.ligand.n_torsions,
            "method": self.method,
            "dl_model_loaded": self.scorer.dl_model is not None,
        }
        report("done", 100, "完成, 用时 {:.1f}s".format(self.stats["elapsed_seconds"]))

        if save_dir is not None:
            self.save(save_dir)
        return results

    # ------------------------------------------------------------ 搜索
    def _search(self, report):
        poses = []
        for i in range(self.exhaustiveness):
            rng = np.random.default_rng(self.seed + 1000 * i + 7)
            use_ga = (self.method == "ga" or
                      (self.method == "hybrid" and i % 2 == 1))
            if use_ga:
                ga = GeneticSearch(self.ligand, self.scorer, self.pocket,
                                   self.box_size, rng,
                                   pop_size=40, n_generations=25)
                out = ga.run()
                # GA 结果再低温细化
                refined = simulated_annealing(
                    self.ligand, self.scorer, self.pocket, self.box_size,
                    rng, n_cycles=3, steps_per_cycle=150, t_init=0.6)
                if refined["score"] < out["score"]:
                    out = refined
            else:
                out = simulated_annealing(
                    self.ligand, self.scorer, self.pocket, self.box_size,
                    rng, n_cycles=8, steps_per_cycle=250)
            poses.append(out)
            pct = 15 + int(60 * (i + 1) / self.exhaustiveness)
            report("search", pct,
                   "运行 {}/{}: 最优能量 {:.2f} kcal/mol".format(
                       i + 1, self.exhaustiveness, out["score"]))
        poses.sort(key=lambda d: d["score"])
        return poses

    # ------------------------------------------------------------ 聚类
    @staticmethod
    def _cluster(poses, cutoff=2.0):
        """贪心聚类: 依能量排序, 吸收 2 A 内构象为代表元簇成员。"""
        reps = []
        for item in poses:
            pose = item["pose"]
            merged = False
            for rep in reps:
                if DockingPipeline._pose_rmsd(pose, rep[0]) <= cutoff:
                    rep[1] += 1
                    merged = True
                    break
            if not merged:
                reps.append([pose, 1])
        return [(p, size) for p, size in reps]

    @staticmethod
    def _pose_rmsd(pose_a: Pose, pose_b: Pose, ligand: Ligand = None,
                   n_sample=40) -> float:
        """两个 pose 的近似 RMSD(按平移距离 + 半径衰减估计, 避免频繁重建构象)。"""
        d_t = float(np.linalg.norm(pose_a.translation - pose_b.translation))
        # 旋转差异: 四元数内积
        dot = abs(float(np.dot(pose_a.rotation_q, pose_b.rotation_q)))
        ang = 2.0 * np.arccos(min(1.0, dot))
        # 经验映射: 平移 + 旋转角(rad)*1.5A 近似 RMSD
        return d_t + 1.5 * ang

    # ------------------------------------------------------------ 输出
    def save(self, save_dir):
        """输出 top 构象 SDF 与 report.json / report 文本。"""
        os.makedirs(save_dir, exist_ok=True)
        for res in self.results:
            out_sdf = os.path.join(save_dir,
                                   "pose_{}.sdf".format(res.rank))
            write_sdf(self.ligand, res.coords, out_sdf,
                      score=res.scores.get("final"), 
                      name="{}_rank{}".format(self.ligand.name, res.rank))
        # 蛋白-配体复合物 PDB(取最优构象)
        if self.results:
            best = self.results[0]
            complex_path = os.path.join(save_dir, "complex_best.pdb")
            with open(complex_path, "w", encoding="utf-8") as fh:
                for i in range(self.protein.n_atoms):
                    x, y, z = self.protein.coords[i]
                    fh.write(
                        "ATOM  {:5d} {:>4s} {:3s} {:1s}{:4d}    "
                        "{:8.3f}{:8.3f}{:8.3f}  1.00  0.00          {:>2s}\n"
                        .format(i + 1, self.protein.atom_names[i],
                                self.protein.res_names[i],
                                self.protein.chains[i],
                                int(self.protein.res_ids[i]), x, y, z,
                                self.protein.elements[i]))
                for j, (xyz, el) in enumerate(zip(best.coords,
                                                  self.ligand.elements)):
                    fh.write("HETATM{:5d} {:>4s} LIG A 999    "
                             "{:8.3f}{:8.3f}{:8.3f}  1.00  0.00          {:>2s}\n"
                             .format(9000 + j, el + str(j + 1),
                                     xyz[0], xyz[1], xyz[2], el))
                fh.write("END\n")
        report = {
            "software": "ProteinDock v1.0",
            "stats": self.stats,
            "results": [r.to_dict() for r in self.results],
        }
        with open(os.path.join(save_dir, "report.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        return save_dir
