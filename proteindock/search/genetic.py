# -*- coding: utf-8 -*-
"""遗传算法(GA)构象搜索

 把配体 pose 编码为「浮点基因」: [平移(3), 四元数(4), 扭转角(T)]。
 采用锦标赛选择 + 精英保留 + 混合交叉(刚体参数凸组合 / 扭转均匀交叉)
 + 自适应高斯变异。与 MC-SA 串联构成混合搜索策略:
 GA 负责全局探索, MC-SA 负责局部细化。
"""

import numpy as np

from ..chemistry.geometry import quaternion_normalize
from .mc_sa import Pose

TOURNAMENT_K = 3        # 锦标赛规模
MUTATE_SIGMA = 0.25     # 变异强度基准


def pose_to_genes(pose: Pose) -> np.ndarray:
    """Pose -> 浮点基因向量 (7+T,)。"""
    return np.concatenate([pose.translation, pose.rotation_q, pose.torsions])


def genes_to_pose(genes: np.ndarray) -> Pose:
    """浮点基因向量 -> Pose。"""
    genes = np.asarray(genes, dtype=np.float64)
    return Pose(genes[:3], genes[3:7], genes[7:])


class GeneticSearch:
    """配体构象搜索的遗传算法实现。"""

    def __init__(self, ligand, scorer, center, box_size, rng,
                 pop_size=48, n_generations=36, elite=4):
        self.ligand = ligand
        self.scorer = scorer
        self.center = np.asarray(center, dtype=np.float64)
        self.box_size = float(box_size)
        self.rng = rng
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.elite = elite
        self.history = []

    # ------------------------------------------------------------ 适应度
    def _fitness(self, pose: Pose) -> float:
        """能量越小适应度问题转化为返回能量(最小化)。"""
        return self.scorer.vina_score(self.ligand, pose.coords(self.ligand))["total"]

    # ------------------------------------------------------------ 算子
    def _init_population(self):
        pop = [Pose.random(self.ligand, self.center, self.box_size, self.rng)
               for _ in range(self.pop_size)]
        return pop

    def _tournament(self, energies):
        """锦标赛选择一个个体下标。"""
        candidates = self.rng.integers(0, self.pop_size, size=TOURNAMENT_K)
        return int(candidates[np.argmin(energies[candidates])])

    def _crossover(self, pa: Pose, pb: Pose) -> Pose:
        """混合交叉: 刚体凸组合 + 扭转均匀交叉。"""
        alpha = self.rng.uniform(0.2, 0.8)
        translation = alpha * pa.translation + (1 - alpha) * pb.translation
        rotation = quaternion_normalize(alpha * pa.rotation_q
                                        + (1 - alpha) * pb.rotation_q)
        if self.ligand.n_torsions:
            mask = self.rng.random(self.ligand.n_torsions) < 0.5
            torsions = np.where(mask, pa.torsions, pb.torsions)
        else:
            torsions = np.zeros(0)
        return Pose(translation, rotation, torsions)

    def _mutate(self, pose: Pose, sigma=MUTATE_SIGMA) -> Pose:
        """自适应高斯变异: 平移 / 旋转 / 扭转各自独立概率扰动。"""
        translation = pose.translation.copy()
        rotation = pose.rotation_q.copy()
        torsions = pose.torsions.copy()
        if self.rng.random() < 0.55:
            translation += self.rng.normal(0, sigma * 2.0, size=3)
        if self.rng.random() < 0.55:
            axis = self.rng.normal(size=3)
            axis /= max(np.linalg.norm(axis), 1e-12)
            ang = self.rng.normal(0, sigma * 1.5)
            half = ang / 2.0
            dq = np.array([np.cos(half), *(np.sin(half) * axis)])
            from ..chemistry.geometry import quaternion_multiply
            rotation = quaternion_multiply(dq, rotation)
        if len(torsions) and self.rng.random() < 0.7:
            k = max(1, int(len(torsions) * 0.35))
            idx = self.rng.choice(len(torsions), size=k, replace=False)
            torsions[idx] += self.rng.normal(0, sigma * 2.5, size=k)
        return Pose(translation, quaternion_normalize(rotation), torsions)

    # ------------------------------------------------------------ 主循环
    def run(self, callback=None):
        """执行进化搜索, 返回最优 pose 与能量。"""
        population = self._init_population()
        energies = np.array([self._fitness(p) for p in population])
        best_i = int(np.argmin(energies))
        best_pose, best_energy = population[best_i].copy(), float(energies[best_i])

        for gen in range(self.n_generations):
            sigma = MUTATE_SIGMA * (1.0 - 0.7 * gen / max(self.n_generations - 1, 1))
            new_pop = []
            # 精英保留
            order = np.argsort(energies)
            for rank in order[:self.elite]:
                new_pop.append(population[rank].copy())
            while len(new_pop) < self.pop_size:
                pa = population[self._tournament(energies)]
                pb = population[self._tournament(energies)]
                child = self._crossover(pa, pb)
                if self.rng.random() < 0.85:
                    child = self._mutate(child, sigma)
                new_pop.append(child)
            population = new_pop
            energies = np.array([self._fitness(p) for p in population])
            gen_best = int(np.argmin(energies))
            if energies[gen_best] < best_energy:
                best_energy = float(energies[gen_best])
                best_pose = population[gen_best].copy()
            self.history.append(float(energies.min()))
            if callback is not None:
                callback(gen, self.n_generations, float(energies.min()),
                         best_energy)

        return {"pose": best_pose, "score": best_energy, "history": self.history}
