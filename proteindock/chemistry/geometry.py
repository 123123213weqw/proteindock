# -*- coding: utf-8 -*-
"""刚体几何运算: 四元数、旋转平移变换与 RMSD

 对接搜索中配体的姿态(pose)由「平移向量 + 单位四元数 + 可旋转键扭转角」
 参数化。使用四元数而非欧拉角可以避免万向节锁, 且数值更稳定。
"""

import numpy as np


# ---------------------------------------------------------------- 四元数工具
def quaternion_identity() -> np.ndarray:
    """单位四元数 (w, x, y, z)。"""
    return np.array([1.0, 0.0, 0.0, 0.0])


def random_quaternion(rng: np.random.Generator) -> np.ndarray:
    """均匀采样 SO(3) 上的单位四元数 (Shoemake 方法)。"""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
        np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
        np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
        np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
    ])
    return q / np.linalg.norm(q)


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """把四元数归一化为单位四元数。"""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return quaternion_identity()
    return q / n


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 q1 * q2。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quaternion_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数旋转向量或向量组。

    :param q: 单位四元数 (w,x,y,z)
    :param v: (3,) 或 (N,3) 坐标数组
    :return: 旋转后同形状坐标数组
    """
    v = np.asarray(v, dtype=np.float64)
    single = v.ndim == 1
    pts = np.atleast_2d(v)
    w, x, y, z = q
    # 旋转矩阵
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    out = pts @ rot.T
    return out[0] if single else out


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """四元数转 3x3 旋转矩阵。"""
    return quaternion_rotate(q, np.eye(3)).T


def small_random_quaternion(rng: np.random.Generator, max_angle_rad: float) -> np.ndarray:
    """在当前姿态附近采样一个随机小旋转(轴均匀, 角度上限给定)。"""
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-12)
    angle = rng.uniform(0.0, max_angle_rad)
    half = angle / 2.0
    q = np.array([np.cos(half), *(np.sin(half) * axis)])
    return quaternion_normalize(q)


# ---------------------------------------------------------------- 刚体变换
class RigidTransform:
    """平移 + 旋转构成的刚体变换, 作用于质心坐标系最为方便。"""

    __slots__ = ("translation", "rotation_q")

    def __init__(self, translation=None, rotation_q=None):
        self.translation = np.zeros(3) if translation is None else np.asarray(translation, float)
        self.rotation_q = quaternion_identity() if rotation_q is None else np.asarray(rotation_q, float)

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """对以质心为原点的坐标组施加变换。"""
        return quaternion_rotate(self.rotation_q, coords) + self.translation

    def compose(self, d_translation: np.ndarray, d_rotation: np.ndarray) -> None:
        """在当前变换上叠加一个小扰动(先旋转后平移)。"""
        self.rotation_q = quaternion_normalize(
            quaternion_multiply(d_rotation, self.rotation_q))
        self.translation = self.translation + d_translation


def centered(coords: np.ndarray) -> np.ndarray:
    """返回以质心为原点的坐标副本。"""
    return coords - coords.mean(axis=0)


# ---------------------------------------------------------------- RMSD
def rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """直接 RMSD(未做叠合), 用于对接构象与晶体构象比较。"""
    diff = np.asarray(coords_a) - np.asarray(coords_b)
    return float(np.sqrt((diff ** 2).sum() / max(len(diff), 1)))


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Kabsch 叠合后最小 RMSD, 忽略整体朝向差异。"""
    Pc = centered(P)
    Qc = centered(Q)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    prot = Pc @ R.T
    return rmsd(prot, Qc)
