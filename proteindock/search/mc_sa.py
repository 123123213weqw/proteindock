# -*- coding: utf-8 -*-
"""蒙特卡洛模拟退火(MC-SA)构象搜索

 配体 pose 由「平移 + 单位四元数 + 可旋转键扭转角」参数化。
 搜索流程借鉴 AutoDock Vina 的 iterated local search 思想:
   随机初始化 -> 高温随机扰动(Metropolis 接受) -> 温度几何降温
   -> 低温局部细化, 并周期性回火(reheat)跳出局部极小。
"""

import numpy as np

from ..chemistry.geometry import (
    RigidTransform,
    quaternion_normalize,
    random_quaternion,
    small_random_quaternion,
)

MAX_TRANSLATE = 2.0     # 单步最大平移 / A
MAX_ROTATE = np.radians(25.0)   # 单步最大旋转
MAX_TORSION = np.radians(30.0)  # 单步最大扭转变化


class Pose:
    """对接构象参数化表示。"""

    __slots__ = ("translation", "rotation_q", "torsions")

    def __init__(self, translation, rotation_q, torsions):
        self.translation = np.asarray(translation, dtype=np.float64)
        self.rotation_q = quaternion_normalize(np.asarray(rotation_q, float))
        self.torsions = np.asarray(torsions, dtype=np.float64)

    # ------------------------------------------------------------ 采样
    @classmethod
    def random(cls, ligand, center, box_size, rng):
        """在口袋中心附近随机初始化 pose(高斯采样, 提高落入率)。"""
        sigma = box_size / 6.0
        half = box_size / 2.0 - 1.0
        t = np.asarray(center) + np.clip(
            rng.normal(0, sigma, size=3), -half, half)
        q = random_quaternion(rng)
        tor = rng.uniform(-np.pi, np.pi, size=ligand.n_torsions)
        return cls(t, q, tor)

    def perturb(self, ligand, rng, temperature=1.0):
        """温度自适应随机扰动, 返回新 Pose。"""
        scale = max(temperature, 0.05)
        dt = rng.uniform(-1, 1, size=3) * MAX_TRANSLATE * scale
        dq = small_random_quaternion(rng, MAX_ROTATE * scale)
        from ..chemistry.geometry import quaternion_multiply
        new = Pose(
            self.translation + dt,
            quaternion_multiply(dq, self.rotation_q),
            self.torsions.copy(),
        )
        if ligand.n_torsions:
            idx = rng.integers(0, ligand.n_torsions)
            new.torsions[idx] += rng.uniform(-1, 1) * MAX_TORSION * scale
        return new

    # ------------------------------------------------------------ 坐标
    def coords(self, ligand) -> np.ndarray:
        """重建该 pose 的配体世界坐标。"""
        tr = RigidTransform(self.translation, self.rotation_q)
        return ligand.conformer(tr)

    def copy(self) -> "Pose":
        return Pose(self.translation.copy(), self.rotation_q.copy(),
                    self.torsions.copy())


def _metropolis(e_old, e_new, temperature, rng):
    """Metropolis 接受准则(能量越低越优)。"""
    if e_new <= e_old:
        return True
    return rng.random() < np.exp(-(e_new - e_old) / max(temperature, 1e-6))


def simulated_annealing(ligand, scorer, center, box_size, rng,
                        n_cycles=8, steps_per_cycle=260,
                        t_init=2.5, t_final=0.02, callback=None):
    """执行一次完整的 MC-SA 搜索。

    :param ligand: io.ligand.Ligand
    :param scorer: scoring.hybrid.HybridScorer (使用其 vina 快速通道)
    :param center: 口袋中心
    :param box_size: 搜索盒子边长 / A
    :param rng: numpy 随机生成器
    :return: dict(pose, score, trajectory)
    """
    pose = Pose.random(ligand, center, box_size, rng)
    energy = scorer.vina_score(ligand, pose.coords(ligand))["total"]
    best_pose, best_energy = pose.copy(), energy
    trajectory = [energy]

    total_steps = n_cycles * steps_per_cycle
    for step in range(total_steps):
        frac = step / max(total_steps - 1, 1)
        # 几何降温 + 周期性回火
        cycle_frac = (step % steps_per_cycle) / steps_per_cycle
        reheat = 1.0 + 0.6 * np.cos(2 * np.pi * (step // steps_per_cycle))
        temperature = (t_init * (t_final / t_init) ** frac) * reheat \
            if cycle_frac > 0 else t_init

        cand = pose.perturb(ligand, rng, temperature)
        cand_energy = scorer.vina_score(ligand, cand.coords(ligand))["total"]
        if _metropolis(energy, cand_energy, temperature, rng):
            pose, energy = cand, cand_energy
            if energy < best_energy:
                best_energy = energy
                best_pose = pose.copy()
        trajectory.append(energy)
        if callback is not None:
            callback(step, total_steps, energy, best_energy)

    return {"pose": best_pose, "score": best_energy, "trajectory": trajectory}
