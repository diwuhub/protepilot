"""
cadet_engine.py  ·  ProtePilot — 生物制药数字孪生系统
==========================================================
CADET-Core  离子交换层析  SMA 模型仿真引擎

版本      : 3.0 (Milestone 1 — 多组分竞争吸附 NCOMP=4)
作者      : Di (ProtePilot)
依赖      : h5py  numpy  matplotlib  (pip install h5py numpy matplotlib)
引擎      : cadet-cli  → engine/cadet-cli.exe  或  engine/cadet-cli

快速开始  :
    python src/cadet_engine.py           # 构建 HDF5 并（若有引擎）运行仿真

────────────────────────────────────────────────────────────────────────
模型结构
────────────────────────────────────────────────────────────────────────
  Unit 000 ─── INLET        分段进料（三次多项式）
      │
      ▼
  Unit 001 ─── LRM + SMA    Lumped-Rate-Model + Steric Mass Action
      │         柱长/孔隙率/流速/弥散系数
      │         吸附: SMA_KA / SMA_KD / SMA_NU / SMA_SIGMA / SMA_LAMBDA
      ▼
  Unit 002 ─── OUTLET       结果导出节点

────────────────────────────────────────────────────────────────────────
四段工艺截面
────────────────────────────────────────────────────────────────────────
  0 ──── 300 ── 600 ──────────────── 2400 ──── 3000  (s)
  │  LOAD   │WASH│   ELUTE (gradient)   │REGEN│
  │  s=50   │ 50 │ 50 → 500 mM lin.grad │ 500 │
  │  p=1    │  0 │         0            │  0  │

────────────────────────────────────────────────────────────────────────
输出目录
────────────────────────────────────────────────────────────────────────
  data/
  ├── <run_name>.h5          CADET 仿真文件（输入 + 输出共用）
  ├── <run_name>_results.csv 出口浓度时序（可选导出）
  └── <run_name>_plot.png    层析图（可选保存）
"""

from __future__ import annotations

import csv
import logging
import os
import platform
import shutil
import subprocess
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

# matplotlib 为可选依赖——仅在绘图时导入
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    # 抑制字体回退警告 & 设置跨平台安全字体
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["font.family"]     = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ─────────────────────────────────────────────────────────────────────────────
# 日志
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CadetEngine")


# ─────────────────────────────────────────────────────────────────────────────
# 内部辅助：HDF5 dataset 写入
# ─────────────────────────────────────────────────────────────────────────────

def _ds(group: h5py.Group, name: str, value) -> None:
    """
    在 HDF5 group 中写入 dataset。

    CADET 规范：所有参数均为 dataset（而非 attribute）。
    此函数统一处理类型转换：
      str          → 定长字节字符串（np.bytes_）
      bool         → np.int32
      int          → np.int32
      float        → np.float64
      list/ndarray → 保留原始 dtype，字符串数组转 bytes
    """
    if isinstance(value, str):
        group.create_dataset(name, data=np.bytes_(value))
    elif isinstance(value, bool):
        group.create_dataset(name, data=np.int32(int(value)))
    elif isinstance(value, int):
        group.create_dataset(name, data=np.int32(value))
    elif isinstance(value, float):
        group.create_dataset(name, data=np.float64(value))
    elif isinstance(value, (list, np.ndarray)):
        arr = np.asarray(value)
        if arr.dtype.kind in ("U", "S", "O"):
            group.create_dataset(name, data=arr.astype("|S"))
        else:
            group.create_dataset(name, data=arr)
    else:
        group.create_dataset(name, data=value)


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProteinParams:
    """
    SMA 吸附模型中蛋白质的物化参数（可由 AI 预测模型提供）。

    Attributes
    ----------
    ka      : 吸附速率常数    [m³/(mol·s)]  典型范围: 1–100
    kd      : 解吸速率常数    [1/s]         典型范围: 100–10000
    lambda_ : 离子容量 Λ      [mol/m³_s]    典型值: 1000–1500
    nu      : 特征电荷数 ν    [–]           典型范围: 2–8
    sigma   : 位阻因子 σ      [–]           典型范围: 5–20
    """
    ka:      float = 35.5
    kd:      float = 1000.0
    lambda_: float = 1200.0
    nu:      float = 4.7
    sigma:   float = 11.83

    @classmethod
    def from_dict(cls, d: Dict) -> "ProteinParams":
        return cls(
            ka      = float(d.get("ka",      cls.ka)),
            kd      = float(d.get("kd",      cls.kd)),
            lambda_ = float(d.get("lambda_", cls.lambda_)),
            nu      = float(d.get("nu",      cls.nu)),
            sigma   = float(d.get("sigma",   cls.sigma)),
        )


@dataclass
class VariantSMA:
    """单个变体（酸性 / 主峰 / 碱性）的 SMA 参数。"""
    ka:    float              # 吸附速率常数  [m³/(mol·s)]
    nu:    float              # 特征电荷数    [–]
    sigma: float              # 位阻因子      [–]

    @classmethod
    def from_dict(cls, d: Dict) -> "VariantSMA":
        return cls(
            ka    = float(d["ka"]),
            nu    = float(d["nu"]),
            sigma = float(d["sigma"]),
        )


@dataclass
class VariantParams:
    """
    多组分电荷变体参数（NCOMP = 4: Salt + Acidic + Main + Basic）。

    SMA 竞争吸附物理:
        dqᵢ/dt = kaᵢ·cᵢ·(Λ − Σ σⱼqⱼ)^νᵢ − kdᵢ·qᵢ·c_salt^νᵢ
        Σ σⱼqⱼ 项自动引入三变体之间的空间位阻竞争。

    Attributes
    ----------
    acidic      : 酸性变体 SMA 参数 (脱酰胺产物 — 负电荷增加 → 弱保留)
    main        : 主峰 (天然构象 mAb)
    basic       : 碱性变体 SMA 参数 (氧化 / 不完全加工 → 强保留)
    kd          : 解吸速率常数 (三变体共享)  [1/s]
    lambda_     : 离子容量 Λ               [mol/m³_s]
    c_fractions : 进样浓度比例 (acidic, main, basic) — 默认 12:80:8
    """
    acidic:      VariantSMA
    main:        VariantSMA
    basic:       VariantSMA
    kd:          float = 1000.0
    lambda_:     float = 1200.0
    c_fractions: Tuple = (0.12, 0.80, 0.08)   # acidic, main, basic

    @classmethod
    def from_dict(cls, d: Dict) -> "VariantParams":
        """
        从字典构造。

        示例:
            VariantParams.from_dict({
                "acidic": {"ka": 25.0, "nu": 3.5, "sigma": 10.0},
                "main":   {"ka": 35.5, "nu": 4.7, "sigma": 11.8},
                "basic":  {"ka": 50.0, "nu": 6.0, "sigma": 13.5},
                "kd": 1000.0,
                "lambda_": 1200.0,
                "c_fractions": [0.12, 0.80, 0.08],
            })
        """
        return cls(
            acidic  = VariantSMA.from_dict(d["acidic"]),
            main    = VariantSMA.from_dict(d["main"]),
            basic   = VariantSMA.from_dict(d["basic"]),
            kd      = float(d.get("kd", 1000.0)),
            lambda_ = float(d.get("lambda_", 1200.0)),
            c_fractions = tuple(d.get("c_fractions", (0.12, 0.80, 0.08))),
        )

    @property
    def variants(self) -> List:
        """按组分顺序返回 [acidic, main, basic]。"""
        return [self.acidic, self.main, self.basic]


@dataclass
class ProcessParams:
    """
    工艺操作参数。

    Attributes
    ----------
    length       : 柱长              [m]      典型值: 0.1–0.5
    col_porosity : 总孔隙率           [–]      典型值: 0.3–0.5
    velocity     : 间隙流速           [m/s]    典型值: 1e-4–1e-3
    flow_rate    : 体积流量           [m³/s]   典型值: 1e-7–1e-5
    col_disp     : 轴向弥散系数       [m²/s]   典型值: 1e-8–1e-6
    c_load       : 蛋白进样浓度       [mol/m³] 典型值: 0.1–5.0
    salt_load    : 上样/洗涤盐浓度    [mol/m³] 典型值: 50
    salt_elute   : 洗脱终止盐浓度     [mol/m³] 典型值: 300–1000
    ncol         : 轴向离散单元数      [–]      典型值: 50–200
    nthreads     : 并行线程数          [–]      默认: 1（conda 版为单线程）
    """
    length:       float = 0.25
    col_porosity: float = 0.37
    velocity:     float = 5.75e-4
    flow_rate:    float = 1.0e-6
    col_disp:     float = 5.75e-8
    c_load:       float = 1.0
    salt_load:    float = 50.0
    salt_elute:   float = 500.0
    gradient_slope: float = 15.0  # <--- 【新增】盐梯度斜率 (mM/min)
    ncol:         int   = 100
    nthreads:     int   = 1

    @classmethod
    def from_dict(cls, d: Dict) -> "ProcessParams":
        return cls(
            length       = float(d.get("length",       cls.length)),
            col_porosity = float(d.get("col_porosity", cls.col_porosity)),
            velocity     = float(d.get("velocity",     cls.velocity)),
            flow_rate    = float(d.get("flow_rate",    cls.flow_rate)),
            col_disp     = float(d.get("col_disp",     cls.col_disp)),
            c_load       = float(d.get("c_load",       cls.c_load)),
            salt_load    = float(d.get("salt_load",    cls.salt_load)),
            salt_elute   = float(d.get("salt_elute",   cls.salt_elute)),
            gradient_slope = float(d.get("gradient_slope", cls.gradient_slope)), # <--- 【新增】
            ncol         = int(d.get("ncol",           cls.ncol)),
            nthreads     = int(d.get("nthreads",       cls.nthreads)),
        )


@dataclass
class SimulationResult:
    """
    仿真完整结果，由 run_simulation() 返回。

    Attributes
    ----------
    h5_path    : 结果所在的 .h5 文件路径（data/ 目录下）
    time       : 时间点数组  [N,]  单位 s
    outlet     : {组分名: 浓度数组}  单位 mol/m³
    returncode : cadet-cli 进程退出码（0 = 成功）
    wall_time  : 仿真实际耗时   [s]
    stdout     : cadet-cli 标准输出
    stderr     : cadet-cli 标准错误
    """
    h5_path:    Path
    time:       np.ndarray
    outlet:     Dict[str, np.ndarray]
    returncode: int
    wall_time:  float
    stdout:     str = ""
    stderr:     str = ""

    # ── 便捷属性 ──────────────────────────────────────────────────────────

    @property
    def salt(self) -> np.ndarray:
        """盐浓度时序曲线 [mol/m³]"""
        return self.outlet.get("Salt", np.array([]))

    @property
    def protein(self) -> np.ndarray:
        """蛋白质浓度时序曲线 [mol/m³]（NCOMP=4 时映射到 Main）"""
        if "Protein" in self.outlet:
            return self.outlet["Protein"]
        return self.outlet.get("Main", np.array([]))

    @property
    def acidic(self) -> np.ndarray:
        """酸性变体浓度时序曲线 [mol/m³]（仅 NCOMP=4 模式）"""
        return self.outlet.get("Acidic", np.array([]))

    @property
    def main(self) -> np.ndarray:
        """主峰浓度时序曲线 [mol/m³]（仅 NCOMP=4 模式）"""
        return self.outlet.get("Main", np.array([]))

    @property
    def basic(self) -> np.ndarray:
        """碱性变体浓度时序曲线 [mol/m³]（仅 NCOMP=4 模式）"""
        return self.outlet.get("Basic", np.array([]))

    @property
    def is_multicomp(self) -> bool:
        """是否为多组分模式"""
        return "Main" in self.outlet

    @property
    def peak_concentration(self) -> float:
        """蛋白出口峰值浓度 [mol/m³]"""
        prot = self.protein
        return float(prot.max()) if len(prot) else 0.0

    @property
    def peak_time(self) -> float:
        """蛋白峰值出现时间 [s]"""
        prot = self.protein
        return float(self.time[prot.argmax()]) if len(prot) else 0.0

    def compute_cqa(self) -> Dict:
        """
        代理方法：调用 CadetEngine.compute_cqa(self)。

        使 result.compute_cqa() 和 engine.compute_cqa(result) 两种
        调用方式均可工作。仅在多组分模式 (NCOMP=4) 下有意义。
        """
        # 延迟导入避免循环引用——CadetEngine 在同一模块中
        engine = CadetEngine.__new__(CadetEngine)
        return engine.compute_cqa(self)

    def summary(self) -> str:
        lines = [
            "═" * 52,
            "  仿真结果摘要",
            "═" * 52,
            f"  输出文件    : {self.h5_path.name}",
            f"  仿真耗时    : {self.wall_time:.2f} s",
            f"  时间点数    : {len(self.time)}",
            f"  蛋白峰值    : {self.peak_concentration:.4f} mol/m³",
            f"  峰值时刻    : {self.peak_time:.1f} s",
            "═" * 52,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────────────────────────────

class CadetEngine:
    """
    CADET-Core 离子交换层析仿真引擎封装。

    典型用法
    --------
    >>> engine = CadetEngine()
    >>> result = engine.run_simulation(
    ...     "my_run.h5",
    ...     prot_params={"ka": 35.5, "kd": 1000, "lambda_": 1200,
    ...                  "nu": 4.7, "sigma": 11.83},
    ...     process_params={"length": 0.25, "velocity": 5.75e-4},
    ... )
    >>> print(result.summary())
    >>> engine.plot_chromatogram(result)
    """

    # ── 四段工艺截面时间节点 (s) ─────────────────────────────────────────
    # sec_000  Load      0 –  300 s   salt = 50 mM,  protein = c_load  ( 5 min)
    # sec_001  Wash    300 –  600 s   salt = 50 mM,  protein = 0       ( 5 min)
    # sec_002  Elute   600 – 2400 s   salt 50 → 500 mM 线性梯度         (30 min)
    # sec_003  Regen  2400 – 3000 s   salt = 500 mM 恒定再生            (10 min)
    #
    # 梯度 30 min（斜率 0.25 mM/s），从 t=600 s 开始。
    # exp_RT 12 min (Humira) → t=720 s → 梯度内 120 s 处 → 可达
    # exp_RT 22 min (Hemlibra) → t=1320 s → 梯度内 720 s 处 → 可达
    # Regen 段确保极强保留蛋白在梯度结束后仍有时间被冲洗出来。
    # 总仿真 50 min (3000 s)，满足所有目标 RT 需求。
    T0          = 0.0       # 开始
    T_LOAD_END  = 300.0     # 上样结束  ( 5 min)
    T_WASH_END  = 600.0     # 洗涤结束  (10 min)
    T_ELUTE_END = 2400.0    # 梯度洗脱结束 (40 min)
    T_HOLD_END  = 3000.0    # 再生结束 / 总仿真时间 (50 min)

    # ── 组分名称 ─────────────────────────────────────────────────────────
    COMP_NAMES:         List[str] = ["Salt", "Protein"]
    COMP_NAMES_VARIANT: List[str] = ["Salt", "Acidic", "Main", "Basic"]

    def __init__(
        self,
        workspace:  str | Path = "data",
        engine_dir: str | Path = "engine",
    ) -> None:
        """
        Parameters
        ----------
        workspace  : .h5 文件与 CSV/PNG 输出目录
        engine_dir : cadet-cli 可执行文件所在目录
        """
        self.workspace  = Path(workspace).resolve()
        self.engine_dir = Path(engine_dir).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        log.info("工作目录 → %s", self.workspace)

    # ═════════════════════════════════════════════════════════════════════
    # ① 公开 API：构建 HDF5
    # ═════════════════════════════════════════════════════════════════════

    def build_h5(
        self,
        filename:       str,
        prot_params:    Union[Dict, ProteinParams, VariantParams],
        process_params: Union[Dict, ProcessParams],
    ) -> Path:
        """
        根据蛋白参数和工艺参数，在 data/ 目录下生成标准 CADET HDF5 文件。

        双模式自动分发:
          - ProteinParams / dict(无 "acidic" 键) → NCOMP=2 单组分模式（向后兼容）
          - VariantParams / dict(含 "acidic" 键) → NCOMP=4 多组分竞争模式

        Parameters
        ----------
        filename       : 输出文件名，例如 "iex_run_001.h5"
        prot_params    : ProteinParams | VariantParams | dict
        process_params : ProcessParams | dict

        Returns
        -------
        Path : 生成的 .h5 文件绝对路径（位于 data/ 目录）
        """
        if isinstance(process_params, dict):
            process_params = ProcessParams.from_dict(process_params)

        # ── 自动检测模式 (Duck Typing — 防止跨模块 isinstance 失效) ──
        if hasattr(prot_params, "main") and hasattr(prot_params, "acidic"):
            # VariantParams 对象 → NCOMP=4 多组分竞争模式
            return self._build_h5_variant(filename, prot_params, process_params)
        elif isinstance(prot_params, dict) and "acidic" in prot_params:
            # dict 含 "acidic" 键 → 转换为 VariantParams → NCOMP=4
            return self._build_h5_variant(
                filename, VariantParams.from_dict(prot_params), process_params)
        else:
            # ProteinParams / 普通 dict → NCOMP=2 单组分模式（向后兼容）
            return self._build_h5_single(filename, prot_params, process_params)

    # ─────────────────────────────────────────────────────────────────
    # build_h5 单组分路径 (NCOMP=2) — 原有逻辑，完全不变
    # ─────────────────────────────────────────────────────────────────

    def _build_h5_single(
        self,
        filename:       str,
        prot_params:    Union[Dict, ProteinParams],
        process_params: ProcessParams,
    ) -> Path:
        if isinstance(prot_params, dict):
            prot_params = ProteinParams.from_dict(prot_params)

        self._validate(prot_params, process_params)

        h5_path = self.workspace / filename

        with h5py.File(h5_path, "w") as f:
            ncomp = len(self.COMP_NAMES)

            # ── /input/model ────────────────────────────────────────────
            model = f.require_group("input/model")
            _ds(model, "NUNITS", np.int32(3))  # INLET + LRM-SMA + OUTLET

            self._write_inlet(model, ncomp, prot_params, process_params)
            self._write_column(model, ncomp, prot_params, process_params)
            self._write_outlet(model, ncomp)
            self._write_connections(model, process_params)

            # ── /input/solver ────────────────────────────────────────────
            self._write_solver(f, process_params)

            # ── /input/return ────────────────────────────────────────────
            self._write_return(f)

            # ── /input/sensitivity ───────────────────────────────────────
            sens = f.require_group("input/sensitivity")
            _ds(sens, "NSENS",       np.int32(0))
            _ds(sens, "SENS_METHOD", "ad1")        # CADET v5 必需，即使 NSENS=0

        log.info("HDF5 构建完成 → %s  (%.1f KB)  NCOMP=2",
                 h5_path.name, h5_path.stat().st_size / 1024)
        return h5_path

    # ─────────────────────────────────────────────────────────────────
    # build_h5 多组分路径 (NCOMP=4) — Milestone 1 新增
    # ─────────────────────────────────────────────────────────────────

    def _build_h5_variant(
        self,
        filename:       str,
        vparams:        VariantParams,
        process_params: ProcessParams,
    ) -> Path:
        """NCOMP=4: Salt + Acidic + Main + Basic 竞争吸附 HDF5 构建。"""
        self._validate_variant(vparams, process_params)

        h5_path = self.workspace / filename
        ncomp = 4
        comp_names = self.COMP_NAMES_VARIANT

        with h5py.File(h5_path, "w") as f:
            model = f.require_group("input/model")
            _ds(model, "NUNITS", np.int32(3))

            self._write_inlet_variant(model, ncomp, vparams, process_params)
            self._write_column_variant(model, ncomp, vparams, process_params)
            self._write_outlet(model, ncomp)
            self._write_connections(model, process_params)
            self._write_solver(f, process_params)
            self._write_return(f)

            sens = f.require_group("input/sensitivity")
            _ds(sens, "NSENS",       np.int32(0))
            _ds(sens, "SENS_METHOD", "ad1")

        log.info("HDF5 构建完成 → %s  (%.1f KB)  NCOMP=4 (多组分竞争)",
                 h5_path.name, h5_path.stat().st_size / 1024)
        return h5_path

    # ═════════════════════════════════════════════════════════════════════
    # ② 公开 API：运行仿真（核心方法）
    # ═════════════════════════════════════════════════════════════════════

    def run_simulation(
        self,
        filename_or_path: str | Path,
        prot_params:       Optional[Dict | ProteinParams] = None,
        process_params:    Optional[Dict | ProcessParams] = None,
        timeout:           int = 600,
        auto_build:        bool = True,
    ) -> SimulationResult:
        """
        通过 subprocess 调用 cadet-cli 执行仿真，结果写回同一 .h5 文件（data/ 目录）。

        Parameters
        ----------
        filename_or_path : .h5 文件名（相对于 data/）或完整路径
        prot_params      : 蛋白 SMA 参数（若文件不存在时自动构建）
        process_params   : 工艺参数（同上）
        timeout          : subprocess 超时时间 [s]，默认 600
        auto_build       : 若 .h5 不存在时自动调用 build_h5()，默认 True

        Returns
        -------
        SimulationResult : 含 time / outlet / h5_path / wall_time 的完整结果对象

        Raises
        ------
        FileNotFoundError : 未找到 .h5 文件 或 cadet-cli 可执行文件
        RuntimeError      : cadet-cli 返回非零退出码
        subprocess.TimeoutExpired : 超过 timeout 秒仍未完成
        """
        # ── 解析路径 ─────────────────────────────────────────────────────
        p = Path(filename_or_path)
        if not p.is_absolute():
            p = self.workspace / p
        if not p.suffix:
            p = p.with_suffix(".h5")

        # ── 按需构建 HDF5 ────────────────────────────────────────────────
        if not p.exists():
            if auto_build and prot_params is not None and process_params is not None:
                log.info("HDF5 不存在，自动构建 …")
                p = self.build_h5(p.name, prot_params, process_params)
            else:
                raise FileNotFoundError(
                    f"仿真文件不存在：{p}\n"
                    "请先调用 build_h5() 或传入 prot_params/process_params 参数。"
                )

        # ── 定位 cadet-cli ───────────────────────────────────────────────
        exe = self._resolve_exe()
        cmd = self._build_cmd(exe, p)

        log.info("启动仿真 ─────────────────────────────────")
        log.info("  引擎  : %s", exe)
        log.info("  命令  : %s", " ".join(cmd))
        log.info("  输入  : %s", p)
        log.info("  超时  : %d s", timeout)

        # ── 构建运行环境（始终注入库路径，确保 cadet-cli 能找到依赖库）────
        run_env = self._build_env()

        # ── subprocess 调用 ──────────────────────────────────────────────
        t_start = _time.monotonic()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env,
            )
        except subprocess.TimeoutExpired:
            raise subprocess.TimeoutExpired(
                cmd=[str(exe), str(p)],
                timeout=timeout,
            )

        wall_time = _time.monotonic() - t_start

        # ── 检查退出码 ───────────────────────────────────────────────────
        if proc.returncode != 0:
            raise RuntimeError(
                f"cadet-cli 仿真失败（退出码 {proc.returncode}，"
                f"耗时 {wall_time:.1f} s）\n"
                f"── STDOUT ──────────────────────────\n{proc.stdout}\n"
                f"── STDERR ──────────────────────────\n{proc.stderr}"
            )

        log.info("仿真完成  ✓  (%.2f s)", wall_time)

        # ── 读取结果 ─────────────────────────────────────────────────────
        t_arr, outlet = self.read_results(p)

        return SimulationResult(
            h5_path    = p,
            time       = t_arr,
            outlet     = outlet,
            returncode = proc.returncode,
            wall_time  = wall_time,
            stdout     = proc.stdout,
            stderr     = proc.stderr,
        )

    # ═════════════════════════════════════════════════════════════════════
    # ③ 公开 API：读取仿真结果（从已有 .h5 中解析）
    # ═════════════════════════════════════════════════════════════════════

    def read_results(
        self,
        h5_path:  str | Path,
        unit_idx: int = 1,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        从已完成仿真的 .h5 文件解析出口浓度时序。

        自动探测组分数:
          - 2 组分 → ["Salt", "Protein"]          (NCOMP=2 向后兼容)
          - 4 组分 → ["Salt", "Acidic", "Main", "Basic"]  (NCOMP=4 竞争模式)

        Returns
        -------
        time   : 时间点数组  shape=(N,)  单位 s
        outlet : {组分名: 浓度数组}       单位 mol/m³
        """
        h5_path = Path(h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"结果文件不存在：{h5_path}")

        outlet: Dict[str, np.ndarray] = {}

        with h5py.File(h5_path, "r") as f:
            time_key = "output/solution/SOLUTION_TIMES"
            if time_key not in f:
                raise KeyError(
                    f"未找到时间轴 [{time_key}]。\n"
                    "可能原因：仿真尚未运行，或 input/return 配置错误。"
                )
            t_arr = f[time_key][:]

            # ── 自动探测组分数 ──────────────────────────────────────────
            base = f"output/solution/unit_{unit_idx:03d}"
            ncomp_detect = 0
            while f"{base}/SOLUTION_OUTLET_COMP_{ncomp_detect:03d}" in f:
                ncomp_detect += 1

            if ncomp_detect == 4:
                comp_names = self.COMP_NAMES_VARIANT
            elif ncomp_detect == 2:
                comp_names = self.COMP_NAMES
            elif ncomp_detect > 0:
                comp_names = [f"Comp_{i}" for i in range(ncomp_detect)]
            else:
                raise KeyError(
                    f"未找到任何组分数据：{base}/SOLUTION_OUTLET_COMP_*\n"
                    f"HDF5 中可用键：{list(f[base].keys()) if base in f else '(路径不存在)'}"
                )

            for i, name in enumerate(comp_names):
                key = f"{base}/SOLUTION_OUTLET_COMP_{i:03d}"
                outlet[name] = f[key][:]

        log.info("结果读取  ✓  %d 个时间点  |  组分 %s  (NCOMP=%d)",
                 len(t_arr), comp_names, ncomp_detect)
        return t_arr, outlet

    # ═════════════════════════════════════════════════════════════════════
    # ④ 公开 API：导出 CSV（保存到 data/ 目录）
    # ═════════════════════════════════════════════════════════════════════

    def export_csv(
        self,
        result:   SimulationResult | Tuple[np.ndarray, Dict],
        stem:     str = "results",
    ) -> Path:
        """
        将出口浓度时序导出为 CSV，保存至 data/<stem>_results.csv。

        Parameters
        ----------
        result : SimulationResult 对象 或 (time, outlet) 元组
        stem   : 输出文件名前缀（不含扩展名）

        Returns
        -------
        Path : 生成的 CSV 文件路径
        """
        if isinstance(result, SimulationResult):
            t_arr, outlet = result.time, result.outlet
        else:
            t_arr, outlet = result

        csv_path = self.workspace / f"{stem}_results.csv"
        # 自动检测组分名（支持 NCOMP=2 和 NCOMP=4）
        comp_names = list(outlet.keys())

        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s"] + comp_names)
            for i, t in enumerate(t_arr):
                row = [f"{t:.6f}"] + [
                    f"{outlet[name][i]:.8e}" for name in comp_names
                ]
                writer.writerow(row)

        log.info("CSV 导出  ✓  → %s  (%d 行)", csv_path.name, len(t_arr))
        return csv_path

    # ═════════════════════════════════════════════════════════════════════
    # ⑤ 公开 API：绘制层析图
    # ═════════════════════════════════════════════════════════════════════

    def plot_chromatogram(
        self,
        result:    SimulationResult | Tuple[np.ndarray, Dict],
        save_path: Optional[str | Path] = None,
        show:      bool = True,
        title:     str  = "IEX-SMA 层析仿真  —  出流曲线",
    ):
        """
        绘制蛋白质出流曲线（上图）+ 盐梯度（下图），并标注工艺截面。

        Parameters
        ----------
        result    : SimulationResult 或 (time, outlet) 元组
        save_path : PNG 保存路径，None 则不保存
        show      : 是否弹出交互窗口
        title     : 图表总标题
        """
        if not _HAS_MPL:
            log.warning("matplotlib 未安装，跳过绘图。pip install matplotlib")
            return None

        if isinstance(result, SimulationResult):
            t_arr, outlet = result.time, result.outlet
        else:
            t_arr, outlet = result

        # ── 布局 ─────────────────────────────────────────────────────────
        fig, (ax_prot, ax_salt) = plt.subplots(
            2, 1,
            figsize=(11, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.05},
        )
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

        PROT_COLOR = "#D32F2F"   # 红
        SALT_COLOR = "#1565C0"   # 蓝
        GRID_KW    = dict(alpha=0.22, linestyle="--", linewidth=0.7)
        SEC_KW     = dict(color="#607D8B", linestyle=":", linewidth=1.0, alpha=0.8)

        # ── 蛋白出流（上图）─────────────────────────────────────────────
        ax_prot.plot(
            t_arr, outlet["Protein"],
            color=PROT_COLOR, linewidth=2.2,
            label="Protein",
        )
        ax_prot.set_ylabel("Protein concentration (mol/m³)", fontsize=10)
        ax_prot.legend(loc="upper right", bbox_to_anchor=(0.99, 0.88), fontsize=9)
        ax_prot.grid(True, **GRID_KW)
        ax_prot.set_xlim(t_arr[0], t_arr[-1])
        ax_prot.set_ylim(bottom=0)

        # ── 盐梯度（下图）────────────────────────────────────────────────
        ax_salt.plot(
            t_arr, outlet["Salt"],
            color=SALT_COLOR, linewidth=1.8,
            linestyle="--", label="Salt",
        )
        ax_salt.set_xlabel("Time (s)", fontsize=10)
        ax_salt.set_ylabel("Salt concentration (mol/m³)", fontsize=10)
        ax_salt.legend(loc="upper left", fontsize=9)
        ax_salt.grid(True, **GRID_KW)
        ax_salt.set_ylim(bottom=0)

        # ── 截面分界线 + 区域标注 ────────────────────────────────────────
        sec_boundaries = [self.T_LOAD_END, self.T_WASH_END, self.T_ELUTE_END]
        sec_labels     = [
            "Load\n(0–300 s)", "Wash\n(300–600 s)",
            "Elute\n(600–2400 s)", "Regen\n(2400–3000 s)",
        ]
        sec_mids       = [
            (self.T0        + self.T_LOAD_END) / 2,
            (self.T_LOAD_END + self.T_WASH_END) / 2,
            (self.T_WASH_END + self.T_ELUTE_END) / 2,
            (self.T_ELUTE_END + self.T_HOLD_END) / 2,
        ]
        sec_colors_bg = ["#E3F2FD", "#F3E5F5", "#E8F5E9", "#FFF3E0"]

        for ax in (ax_prot, ax_salt):
            # 背景色块
            boundaries = [self.T0] + sec_boundaries + [self.T_HOLD_END]
            for i in range(4):
                ax.axvspan(
                    boundaries[i], boundaries[i + 1],
                    color=sec_colors_bg[i], alpha=0.18, zorder=0,
                )
            # 分界线
            for tb in sec_boundaries:
                ax.axvline(tb, **SEC_KW)

        # 截面名称标注（仅在上图顶部）
        y_top = ax_prot.get_ylim()[1]
        for mid, lbl in zip(sec_mids, sec_labels):
            ax_prot.text(
                mid, y_top * 0.97, lbl,
                ha="center", va="top", fontsize=7.5,
                color="#37474F",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          alpha=0.75, ec="none"),
            )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            save_path = Path(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("图表保存  ✓  → %s", save_path)

        if show:
            plt.show()

        return fig

    # ═════════════════════════════════════════════════════════════════════
    # ⑥ 公开 API：一键完整流程
    # ═════════════════════════════════════════════════════════════════════

    def run_all(
        self,
        filename:       str,
        prot_params:    Dict | ProteinParams,
        process_params: Dict | ProcessParams,
        export_csv:     bool = True,
        save_plot:      bool = True,
        timeout:        int  = 600,
    ) -> SimulationResult:
        """
        完整流程：build_h5 → run_simulation → export_csv → plot_chromatogram。

        输出全部写入 data/ 目录：
          data/<filename>.h5
          data/<stem>_results.csv   (若 export_csv=True)
          data/<stem>_plot.png      (若 save_plot=True)
        """
        stem = Path(filename).stem

        # 1. 构建 + 仿真
        result = self.run_simulation(
            filename,
            prot_params    = prot_params,
            process_params = process_params,
            timeout        = timeout,
            auto_build     = True,
        )

        # 2. 导出 CSV
        if export_csv:
            self.export_csv(result, stem=stem)

        # 3. 绘图
        if save_plot:
            png_path = self.workspace / f"{stem}_plot.png"
            self.plot_chromatogram(result, save_path=png_path, show=False)

        log.info("%s", result.summary())
        return result

    # ═════════════════════════════════════════════════════════════════════
    # ⑦ 公开 API：CQA 分离度分析 (Milestone 1)
    # ═════════════════════════════════════════════════════════════════════

    def compute_cqa(
        self,
        result: Union[SimulationResult, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    ) -> Dict:
        """
        从多组分仿真结果计算 CQA 分离质量指标。

        分离度公式 (USP Resolution):
            Rs = 2 · (tR2 − tR1) / (w1 + w2)
        其中 w 为基线峰宽，由半高全宽 (FWHM) 近似:
            w ≈ 1.699 × FWHM   (高斯峰假设)

        Parameters
        ----------
        result : SimulationResult 或 (time, outlet) 元组
                 必须含有 "Acidic", "Main", "Basic" 组分

        Returns
        -------
        dict : {
            "peaks": {
                "Acidic": {"rt_min", "fwhm_min", "height", "area"},
                "Main":   {...},
                "Basic":  {...},
            },
            "resolution": {
                "Acidic_vs_Main": float,
                "Main_vs_Basic":  float,
            },
            "area_pct": {
                "Acidic": float,
                "Main":   float,
                "Basic":  float,
            },
        }
        """
        if isinstance(result, SimulationResult):
            t_arr, outlet = result.time, result.outlet
        else:
            t_arr, outlet = result

        required = ["Acidic", "Main", "Basic"]
        for name in required:
            if name not in outlet:
                raise KeyError(
                    f"compute_cqa 需要多组分数据（缺少 '{name}'）。"
                    f"当前组分: {list(outlet.keys())}"
                )

        t_min = t_arr / 60.0   # s → min

        # ── 各组分峰检测 ─────────────────────────────────────────────
        peaks = {}
        for name in required:
            conc = outlet[name]
            peaks[name] = self._detect_peak(t_min, conc)

        # ── 面积百分比 ───────────────────────────────────────────────
        total_area = sum(p["area"] for p in peaks.values())
        area_pct = {}
        for name in required:
            area_pct[name] = (peaks[name]["area"] / total_area * 100.0
                              if total_area > 0 else 0.0)

        # ── 分离度计算 ───────────────────────────────────────────────
        resolution = {}
        for name_a, name_b, label in [
            ("Acidic", "Main",  "Acidic_vs_Main"),
            ("Main",   "Basic", "Main_vs_Basic"),
        ]:
            pa, pb = peaks[name_a], peaks[name_b]
            # 基线峰宽 ≈ 1.699 × FWHM (高斯近似)
            w_a = pa["fwhm_min"] * 1.699
            w_b = pb["fwhm_min"] * 1.699
            denom = w_a + w_b
            if denom > 0 and pa["rt_min"] > 0 and pb["rt_min"] > 0:
                resolution[label] = 2.0 * abs(pb["rt_min"] - pa["rt_min"]) / denom
            else:
                resolution[label] = 0.0

        return {
            "peaks":      peaks,
            "resolution": resolution,
            "area_pct":   area_pct,
        }

    @staticmethod
    def _detect_peak(
        t_min: np.ndarray,
        conc:  np.ndarray,
    ) -> Dict:
        """
        单峰检测：从浓度曲线提取 RT、FWHM、峰高、面积。

        Returns
        -------
        dict : {"rt_min", "fwhm_min", "height", "area"}
        """
        # 峰值
        peak_idx = int(np.argmax(conc))
        height   = float(conc[peak_idx])
        rt_min   = float(t_min[peak_idx])

        if height <= 0:
            return {"rt_min": 0.0, "fwhm_min": 0.0, "height": 0.0, "area": 0.0}

        # FWHM: 半高位置的左右插值
        half_h = height / 2.0

        # 左侧半高点
        left_idx = peak_idx
        while left_idx > 0 and conc[left_idx] > half_h:
            left_idx -= 1
        if left_idx < peak_idx and conc[left_idx] < half_h:
            # 线性插值精确位置
            frac = (half_h - conc[left_idx]) / max(conc[left_idx + 1] - conc[left_idx], 1e-30)
            t_left = t_min[left_idx] + frac * (t_min[left_idx + 1] - t_min[left_idx])
        else:
            t_left = t_min[left_idx]

        # 右侧半高点
        right_idx = peak_idx
        while right_idx < len(conc) - 1 and conc[right_idx] > half_h:
            right_idx += 1
        if right_idx > peak_idx and conc[right_idx] < half_h:
            frac = (half_h - conc[right_idx]) / max(conc[right_idx - 1] - conc[right_idx], 1e-30)
            t_right = t_min[right_idx] - frac * (t_min[right_idx] - t_min[right_idx - 1])
        else:
            t_right = t_min[right_idx]

        fwhm = max(t_right - t_left, 0.0)

        # 面积 (梯形积分，仅积分显著区域)
        _trapz_fn = getattr(np, "trapezoid", None) or np.trapz
        area = float(_trapz_fn(conc, t_min))

        return {
            "rt_min":   round(rt_min, 4),
            "fwhm_min": round(fwhm, 4),
            "height":   round(height, 6),
            "area":     round(area, 6),
        }

    # ═════════════════════════════════════════════════════════════════════
    # 私有：Unit 000  INLET
    # ═════════════════════════════════════════════════════════════════════

    def _write_inlet(
        self,
        model:   h5py.Group,
        ncomp:   int,
        prot:    ProteinParams,
        proc:    ProcessParams,
    ) -> None:
        """
        进料单元（Unit 000）— 分段三次多项式（PIECEWISE_CUBIC_POLY）。

        截面进料浓度分配
        ─────────────────────────────────────────────────────────
          sec_000  Load   0– 300 s  salt=c_salt_load  protein=c_load
          sec_001  Wash  300– 600 s  salt=c_salt_load  protein=0
          sec_002  Elute 600–2400 s  salt 线性增至 salt_elute  protein=0
          sec_003  Regen 2400–3000 s salt=salt_elute  protein=0
        ─────────────────────────────────────────────────────────
        LIN_COEFF[salt] = (salt_elute - salt_load) / (T_ELUTE_END - T_WASH_END)
                        = (500 - 50) / 1800 = 0.25  mol/(m³·s)
        """
        u = model.require_group("unit_000")
        _ds(u, "UNIT_TYPE",  "INLET")
        _ds(u, "NCOMP",      np.int32(ncomp))
        _ds(u, "INLET_TYPE", "PIECEWISE_CUBIC_POLY")

        salt_slope = (proc.salt_elute - proc.salt_load) / (
            self.T_ELUTE_END - self.T_WASH_END
        )

        # ── 每截面的 4 阶多项式系数 (CONST / LIN / QUAD / CUBE) ─────────
        # 格式: [salt_coeff, protein_coeff]
        # 【修改】动态计算洗脱时间与每秒斜率
        elute_duration_s = (proc.salt_elute - proc.salt_load) / proc.gradient_slope * 60.0
        salt_slope_sec = proc.gradient_slope / 60.0  # mM/s
        
        sections = {
            "sec_000": dict(  # Load
                CONST_COEFF = np.array([proc.salt_load, proc.c_load], np.float64),
                LIN_COEFF   = np.zeros(ncomp, np.float64),
            ),
            "sec_001": dict(  # Wash
                CONST_COEFF = np.array([proc.salt_load, 0.0], np.float64),
                LIN_COEFF   = np.zeros(ncomp, np.float64),
            ),
            "sec_002": dict(  # Elute (gradient)
                CONST_COEFF = np.array([proc.salt_load, 0.0], np.float64),
                LIN_COEFF   = np.array([salt_slope_sec, 0.0],     np.float64),
            ),
            "sec_003": dict(  # Regen (恒定高盐再生)
                CONST_COEFF = np.array([proc.salt_elute, 0.0], np.float64),
                LIN_COEFF   = np.zeros(ncomp, np.float64),
            ),
        }

        # CADET v5: sections 直接在 unit_000/ 下，无 inlet/ 子组
        for sec_name, coeffs in sections.items():
            sg = u.require_group(sec_name)
            _ds(sg, "CONST_COEFF", coeffs["CONST_COEFF"])
            _ds(sg, "LIN_COEFF",   coeffs["LIN_COEFF"])
            _ds(sg, "QUAD_COEFF",  np.zeros(ncomp, np.float64))
            _ds(sg, "CUBE_COEFF",  np.zeros(ncomp, np.float64))

    # ═════════════════════════════════════════════════════════════════════
    # 私有：Unit 001  LRM 柱 + SMA 吸附
    # ═════════════════════════════════════════════════════════════════════

    def _write_column(
        self,
        model: h5py.Group,
        ncomp: int,
        prot:  ProteinParams,
        proc:  ProcessParams,
    ) -> None:
        """
        层析柱单元（Unit 001）— Lumped Rate Model Without Pores (LRM)。

        LRM 适用场景：颗粒内传质阻力可忽略（高流速或小颗粒）。
        若需要精确颗粒内扩散建模，应改用 GENERAL_RATE_MODEL (GRM)。

        必需参数列表
        ─────────────────────────────────────────────────────────
          COL_LENGTH     柱长
          COL_POROSITY   总孔隙率（εt = 柱间 + 颗粒内）
          VELOCITY       间隙流速
          COL_DISPERSION 轴向弥散系数
          INIT_C         初始流动相浓度向量
          INIT_Q         初始结合相浓度向量
          adsorption/    SMA 吸附子模型
          discretization/ 有限体积离散化
        """
        col = model.require_group("unit_001")
        _ds(col, "UNIT_TYPE", "LUMPED_RATE_MODEL_WITHOUT_PORES")
        _ds(col, "NCOMP",     np.int32(ncomp))

        # NBOUND: 每组分的吸附态数量
        #   组分 0（盐/反离子）: 1 个结合态（SMA 守恒约束）
        #   组分 1（蛋白）    : 1 个结合态
        _ds(col, "NBOUND", np.array([1, 1], dtype=np.int32))

        # ── 柱物理参数 ──────────────────────────────────────────────────
        # LRM Without Pores 使用 TOTAL_POROSITY（非 COL_POROSITY）
        _ds(col, "COL_LENGTH",      np.float64(proc.length))
        _ds(col, "TOTAL_POROSITY",  np.float64(proc.col_porosity))
        _ds(col, "VELOCITY",        np.float64(proc.velocity))
        _ds(col, "COL_DISPERSION",  np.float64(proc.col_disp))

        # ── 初始条件 ────────────────────────────────────────────────────
        # INIT_C: 初始流动相浓度向量
        #   [salt_load, 0.0]  → 柱预平衡在 salt_load 盐浓度，无蛋白
        # INIT_Q: 初始结合相浓度向量
        #   [lambda_, 0.0]    → 盐占满所有离子交换位点（SMA 电荷守恒：q_s = Λ）
        _ds(col, "INIT_C", np.array([proc.salt_load, 0.0], np.float64))
        _ds(col, "INIT_Q", np.array([prot.lambda_,   0.0], np.float64))

        # CADET v5: ADSORPTION_MODEL 在 unit_001/ 层级，不在 adsorption/ 子组
        _ds(col, "ADSORPTION_MODEL", "STERIC_MASS_ACTION")

        # ── SMA 吸附子模型 ───────────────────────────────────────────────
        self._write_sma(col, ncomp, prot)

        # ── 数值离散化 ───────────────────────────────────────────────────
        self._write_discretization(col, proc.ncol)

    def _write_sma(
        self,
        col:   h5py.Group,
        ncomp: int,
        prot:  ProteinParams,
    ) -> None:
        """
        SMA（Steric Mass Action）吸附模型参数。

        SMA 吸附速率方程（组分 i）:
            dqᵢ/dt = kaᵢ·cᵢ·(Λ - Σσⱼqⱼ)^νᵢ − kdᵢ·qᵢ·c_salt^νᵢ

        离子守恒约束:
            q_salt = Λ − Σ(νᵢ·qᵢ)   （盐不具有吸附动力学，由约束方程决定）

        组分 0（盐）: ka=kd=nu=sigma=0（不遵循动力学，由守恒约束隐式求解）
        组分 1（蛋白）: 使用 prot 中的参数
        """
        ads = col.require_group("adsorption")

        # ADSORPTION_MODEL 已在 unit_001/ 层级声明（_write_column 中）
        _ds(ads, "IS_KINETIC", np.int32(1))   # 1 = 动力学  0 = 拟稳态

        _ds(ads, "SMA_LAMBDA", np.float64(prot.lambda_))
        _ds(ads, "SMA_KA",     np.array([0.0,       prot.ka],    np.float64))
        _ds(ads, "SMA_KD",     np.array([0.0,       prot.kd],    np.float64))
        _ds(ads, "SMA_NU",     np.array([0.0,       prot.nu],    np.float64))
        _ds(ads, "SMA_SIGMA",  np.array([0.0,       prot.sigma], np.float64))

    # ═════════════════════════════════════════════════════════════════════
    # Milestone 1: 多组分 NCOMP=4 私有方法
    # ═════════════════════════════════════════════════════════════════════

    def _write_inlet_variant(
        self,
        model:   h5py.Group,
        ncomp:   int,
        vp:      VariantParams,
        proc:    ProcessParams,
    ) -> None:
        """
        NCOMP=4 进料单元 — 组分顺序: [Salt, Acidic, Main, Basic]。

        进样段 (Load): 三变体按 c_fractions 分配 c_load 浓度
        洗涤/洗脱/再生: 仅盐变化，蛋白浓度为 0
        """
        u = model.require_group("unit_000")
        _ds(u, "UNIT_TYPE",  "INLET")
        _ds(u, "NCOMP",      np.int32(ncomp))
        _ds(u, "INLET_TYPE", "PIECEWISE_CUBIC_POLY")

        # 梯度斜率 (mol/(m³·s))
        salt_slope_sec = proc.gradient_slope / 60.0  # mM/min → mol/(m³·s)

        # 蛋白进样浓度分配
        c_acid = proc.c_load * vp.c_fractions[0]
        c_main = proc.c_load * vp.c_fractions[1]
        c_base = proc.c_load * vp.c_fractions[2]

        sections = {
            "sec_000": dict(  # Load
                CONST_COEFF = np.array([proc.salt_load, c_acid, c_main, c_base],
                                       np.float64),
                LIN_COEFF   = np.zeros(ncomp, np.float64),
            ),
            "sec_001": dict(  # Wash
                CONST_COEFF = np.array([proc.salt_load, 0.0, 0.0, 0.0],
                                       np.float64),
                LIN_COEFF   = np.zeros(ncomp, np.float64),
            ),
            "sec_002": dict(  # Elute (gradient)
                CONST_COEFF = np.array([proc.salt_load, 0.0, 0.0, 0.0],
                                       np.float64),
                LIN_COEFF   = np.array([salt_slope_sec, 0.0, 0.0, 0.0],
                                       np.float64),
            ),
            "sec_003": dict(  # Regen
                CONST_COEFF = np.array([proc.salt_elute, 0.0, 0.0, 0.0],
                                       np.float64),
                LIN_COEFF   = np.zeros(ncomp, np.float64),
            ),
        }

        for sec_name, coeffs in sections.items():
            sg = u.require_group(sec_name)
            _ds(sg, "CONST_COEFF", coeffs["CONST_COEFF"])
            _ds(sg, "LIN_COEFF",   coeffs["LIN_COEFF"])
            _ds(sg, "QUAD_COEFF",  np.zeros(ncomp, np.float64))
            _ds(sg, "CUBE_COEFF",  np.zeros(ncomp, np.float64))

    def _write_column_variant(
        self,
        model: h5py.Group,
        ncomp: int,
        vp:    VariantParams,
        proc:  ProcessParams,
    ) -> None:
        """
        NCOMP=4 层析柱单元 — LRM + SMA 多组分竞争吸附。

        NBOUND = [1, 1, 1, 1]: 盐(SMA守恒) + 3变体各1个结合态
        INIT_C = [salt_load, 0, 0, 0]: 柱预平衡
        INIT_Q = [lambda_, 0, 0, 0]:   盐占满交换位点
        """
        col = model.require_group("unit_001")
        _ds(col, "UNIT_TYPE", "LUMPED_RATE_MODEL_WITHOUT_PORES")
        _ds(col, "NCOMP",     np.int32(ncomp))

        # NBOUND: 每组分 1 个结合态
        _ds(col, "NBOUND", np.array([1, 1, 1, 1], dtype=np.int32))

        # 柱物理参数
        _ds(col, "COL_LENGTH",      np.float64(proc.length))
        _ds(col, "TOTAL_POROSITY",  np.float64(proc.col_porosity))
        _ds(col, "VELOCITY",        np.float64(proc.velocity))
        _ds(col, "COL_DISPERSION",  np.float64(proc.col_disp))

        # 初始条件
        _ds(col, "INIT_C", np.array([proc.salt_load, 0.0, 0.0, 0.0], np.float64))
        _ds(col, "INIT_Q", np.array([vp.lambda_,     0.0, 0.0, 0.0], np.float64))

        _ds(col, "ADSORPTION_MODEL", "STERIC_MASS_ACTION")

        # SMA 参数
        self._write_sma_variant(col, ncomp, vp)

        # 离散化
        self._write_discretization(col, proc.ncol)

    def _write_sma_variant(
        self,
        col:   h5py.Group,
        ncomp: int,
        vp:    VariantParams,
    ) -> None:
        """
        NCOMP=4 SMA 吸附参数配置。

        组分顺序: [Salt(0), Acidic(1), Main(2), Basic(3)]

        竞争排阻物理:
          dqᵢ/dt = kaᵢ·cᵢ·(Λ − Σⱼ σⱼ·qⱼ)^νᵢ − kdᵢ·qᵢ·c_salt^νᵢ
          其中 Σⱼ σⱼ·qⱼ 同时包含 Acidic + Main + Basic 的位阻贡献。
          → 三变体在有限离子容量 Λ 上竞争结合位点。
        """
        ads = col.require_group("adsorption")
        _ds(ads, "IS_KINETIC", np.int32(1))

        _ds(ads, "SMA_LAMBDA", np.float64(vp.lambda_))

        # 组分 0 (Salt): SMA 守恒约束，ka=kd=nu=sigma=0
        _ds(ads, "SMA_KA", np.array(
            [0.0, vp.acidic.ka, vp.main.ka, vp.basic.ka], np.float64))
        _ds(ads, "SMA_KD", np.array(
            [0.0, vp.kd,        vp.kd,      vp.kd],       np.float64))
        _ds(ads, "SMA_NU", np.array(
            [0.0, vp.acidic.nu, vp.main.nu, vp.basic.nu], np.float64))
        _ds(ads, "SMA_SIGMA", np.array(
            [0.0, vp.acidic.sigma, vp.main.sigma, vp.basic.sigma],
            np.float64))

    @staticmethod
    def _validate_variant(vp: VariantParams, proc: ProcessParams) -> None:
        """校验多组分参数合理性。"""
        errors: List[str] = []

        for name, var in [("acidic", vp.acidic),
                          ("main", vp.main),
                          ("basic", vp.basic)]:
            if var.ka <= 0:
                errors.append(f"{name}.ka={var.ka} 必须 > 0")
            if var.nu <= 0:
                errors.append(f"{name}.nu={var.nu} 必须 > 0")
            if var.sigma < 0:
                errors.append(f"{name}.sigma={var.sigma} 不应为负")

        if vp.kd <= 0:
            errors.append(f"kd={vp.kd} 必须 > 0")
        if vp.lambda_ <= 0:
            errors.append(f"lambda_={vp.lambda_} 必须 > 0")

        frac_sum = sum(vp.c_fractions)
        if abs(frac_sum - 1.0) > 0.01:
            errors.append(f"c_fractions 之和 = {frac_sum:.3f} ≠ 1.0")

        if not (0 < proc.col_porosity < 1):
            errors.append(f"col_porosity={proc.col_porosity} 必须在 (0, 1)")
        if proc.length <= 0:
            errors.append(f"length={proc.length} 必须 > 0")
        if proc.velocity <= 0:
            errors.append(f"velocity={proc.velocity} 必须 > 0")
        if proc.salt_elute <= proc.salt_load:
            errors.append(
                f"salt_elute({proc.salt_elute}) 必须 > salt_load({proc.salt_load})")

        if errors:
            raise ValueError("多组分参数校验失败：\n  " + "\n  ".join(errors))

    # ═════════════════════════════════════════════════════════════════════

    def _write_discretization(self, col: h5py.Group, ncol: int = 100) -> None:
        """
        有限体积空间离散化 + WENO 高阶重构。

        WENO_ORDER = 3  对应 5 阶精度重构方案，在保持数值稳定性的同时
        有效抑制数值色散（数值扩散），适合层析峰形仿真。
        """
        disc = col.require_group("discretization")
        _ds(disc, "NCOL",                  np.int32(ncol))
        _ds(disc, "NPARTYPE",              np.int32(0))    # LRM: 0 颗粒类型
        _ds(disc, "USE_ANALYTIC_JACOBIAN", np.int32(1))

        weno = disc.require_group("weno")
        _ds(weno, "WENO_ORDER",     np.int32(3))        # 3 = 5th-order
        _ds(weno, "BOUNDARY_MODEL", np.int32(0))        # 0 = 零通量左边界
        _ds(weno, "WENO_EPS",       np.float64(1e-10))  # 平滑性指标下限

    # ═════════════════════════════════════════════════════════════════════
    # 私有：Unit 002  OUTLET
    # ═════════════════════════════════════════════════════════════════════

    def _write_outlet(self, model: h5py.Group, ncomp: int) -> None:
        """
        出口单元（Unit 002）。

        CADET 必须有 OUTLET 节点才能将 unit_001 的出口浓度写入
        output/solution/unit_001/SOLUTION_OUTLET_COMP_XXX。
        """
        u = model.require_group("unit_002")
        _ds(u, "UNIT_TYPE", "OUTLET")
        _ds(u, "NCOMP",     np.int32(ncomp))

    # ═════════════════════════════════════════════════════════════════════
    # 私有：流路连接网络
    # ═════════════════════════════════════════════════════════════════════

    def _write_connections(
        self,
        model: h5py.Group,
        proc:  ProcessParams,
    ) -> None:
        """
        单元流路拓扑 — INLET(0) → LRM柱(1) → OUTLET(2)。

        CONNECTIONS 矩阵布局（每行 5 个元素）:
            [from_unit, to_unit, from_port, to_port, flow_rate]
            端口 -1 表示"单端口单元的全部端口"

        NSWITCHES = 1：全程只有一个流路配置（Load/Wash/Elute 期间流路不切换）。
        SECTION   = 0：从第 0 截面（t=0）开始生效。
        """
        # CADET v5: connections 在 /input/model/connections/
        conn = model.require_group("connections")
        _ds(conn, "NSWITCHES", np.int32(1))

        sw = conn.require_group("switch_000")
        _ds(sw, "SECTION", np.int32(0))
        _ds(sw, "CONNECTIONS", np.array([
            0, 1, -1, -1, proc.flow_rate,   # INLET  → LRM 柱
            1, 2, -1, -1, proc.flow_rate,   # LRM 柱 → OUTLET
        ], dtype=np.float64))

    # ═════════════════════════════════════════════════════════════════════
    # 私有：IDAS 求解器配置
    # ═════════════════════════════════════════════════════════════════════

    def _write_solver(self, f: h5py.File, proc: ProcessParams) -> None:

        # 【修改】根据真实实验斜率，动态推算洗脱结束点和总时长
        elute_duration_s = (proc.salt_elute - proc.salt_load) / proc.gradient_slope * 60.0
        t_elute_end = self.T_WASH_END + elute_duration_s
        t_hold_end = t_elute_end + 600.0  # 保证洗脱结束后有 10 分钟稳定期

        """
        IDAS 时间积分器参数 + 四段截面时间表。

        截面时间节点（Section Times）:
            [0, 300, 600, 2400, 3000]  单位 s
             └──Load──┘└Wash┘└───Elute────┘└Regen┘
               600 s   300 s    1800 s     300 s

        SECTION_CONTINUITY = [0, 0, 0]:
            每次截面切换均为阶跃。
        """
        slv = f.require_group("input/solver")
        _ds(slv, "NTHREADS", np.int32(proc.nthreads))

        # CADET v5: /input/model/solver — 线性求解器（Krylov/GMRES）参数
        mslv = f.require_group("input/model/solver")
        _ds(mslv, "MAX_KRYLOV",    np.int32(0))      # 0 = 自动（min(NDOF,30)）
        _ds(mslv, "GS_TYPE",       np.int32(1))      # 1 = 经典 Gram-Schmidt
        _ds(mslv, "MAX_RESTARTS",  np.int32(10))
        _ds(mslv, "SCHUR_SAFETY",  np.float64(1e-8))

        # 均匀输出时间点（1 500 点覆盖完整 50 min → 2.0 s 间隔）
        _ds(slv, "USER_SOLUTION_TIMES",
            np.linspace(self.T0, t_hold_end, 1500, dtype=np.float64))

        # 时间积分器
        # ── 容差说明 ──────────────────────────────────────────────────
        # ALGTOL 1e-12 过于严苛，导致 IDA 在截面切换时步长缩至 ~1e-13
        # 触发 IDA_CONV_FAIL。对于层析仿真，1e-6 完全足够。
        # MAX_STEP_SIZE 限制最大步长，防止跨越截面边界时积分跳变。
        # CONSISTENT_INIT_MODE = 1: 每次截面切换后重新做一致性初始化
        #   （修复 Load→Wash 阶跃引起的 DAE 不一致问题）
        ti = slv.require_group("time_integrator")
        _ds(ti, "ABSTOL",              np.float64(1e-6))
        _ds(ti, "RELTOL",             np.float64(1e-5))
        _ds(ti, "ALGTOL",             np.float64(1e-6))
        _ds(ti, "INIT_STEP_SIZE",     np.float64(1e-4))
        _ds(ti, "MAX_STEP_SIZE",      np.float64(10.0))
        _ds(ti, "MAX_STEPS",          np.int32(5_000_000))

        # 一致性初始化：截面切换时重新计算 DAE 一致初值
        # 1 = Full (y + yd)，对阶跃式 Load→Wash 切换至关重要
        _ds(ti, "CONSISTENT_INIT_MODE",      np.int32(1))
        _ds(ti, "CONSISTENT_INIT_MODE_SENS", np.int32(1))

        # 截面时间表：CADET v5 要求在 /input/solver/sections/
        # 4 段：Load | Wash | Elute (gradient) | Regen (恒定高盐再生)
        sec = slv.require_group("sections")
        _ds(sec, "NSEC",              np.int32(4))
        _ds(sec, "SECTION_TIMES",     np.array(
            [self.T0, self.T_LOAD_END, self.T_WASH_END, t_elute_end, t_hold_end],
            dtype=np.float64,
        ))
        # SECTION_CONTINUITY: 每个截面边界的连续性标志 (NSEC-1 个)
        _ds(sec, "SECTION_CONTINUITY", np.array([0, 0, 0], dtype=np.int32))

    # ═════════════════════════════════════════════════════════════════════
    # 私有：输出控制
    # ═════════════════════════════════════════════════════════════════════

    def _write_return(self, f: h5py.File) -> None:
        """
        控制 CADET 将哪些信号写入 output/ 组。

        仅开启 WRITE_SOLUTION_OUTLET：记录柱出口浓度（节省磁盘空间）。
        SPLIT_COMPONENTS_DATA = 1：每个组分独立 dataset，便于后处理。
        """
        ret = f.require_group("input/return/unit_001")
        _ds(ret, "WRITE_SOLUTION_OUTLET",   np.int32(1))
        _ds(ret, "WRITE_SOLUTION_INLET",    np.int32(0))
        _ds(ret, "WRITE_SOLUTION_BULK",     np.int32(0))
        _ds(ret, "WRITE_SOLUTION_PARTICLE", np.int32(0))
        _ds(ret, "WRITE_SOLUTION_FLUX",     np.int32(0))
        _ds(ret, "WRITE_SENS_OUTLET",       np.int32(0))
        _ds(ret, "SPLIT_COMPONENTS_DATA",   np.int32(1))

    # ═════════════════════════════════════════════════════════════════════
    # 私有：参数校验
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _validate(prot: ProteinParams, proc: ProcessParams) -> None:
        """校验关键参数合理性，提前发现常见配置错误。"""
        errors: List[str] = []

        if prot.ka <= 0:
            errors.append(f"ka={prot.ka} 必须 > 0")
        if prot.kd <= 0:
            errors.append(f"kd={prot.kd} 必须 > 0")
        if prot.lambda_ <= 0:
            errors.append(f"lambda_={prot.lambda_} 必须 > 0")
        if prot.nu <= 0:
            errors.append(f"nu={prot.nu} 必须 > 0")
        if not (0 < proc.col_porosity < 1):
            errors.append(f"col_porosity={proc.col_porosity} 必须在 (0, 1)")
        if proc.length <= 0:
            errors.append(f"length={proc.length} 必须 > 0")
        if proc.velocity <= 0:
            errors.append(f"velocity={proc.velocity} 必须 > 0")
        if proc.salt_elute <= proc.salt_load:
            errors.append(
                f"salt_elute({proc.salt_elute}) 必须 > salt_load({proc.salt_load})"
            )

        if errors:
            raise ValueError("参数校验失败：\n  " + "\n  ".join(errors))

    # ═════════════════════════════════════════════════════════════════════
    # ⑦ 公开 API：引擎环境诊断（排查 dylib / .so 缺失问题）
    # ═════════════════════════════════════════════════════════════════════

    def check_engine(self) -> bool:
        """
        检查 cadet-cli 及其动态库依赖是否齐全，并打印诊断报告。

        Returns
        -------
        bool : True = 一切就绪；False = 存在缺失，需要手动修复

        常见错误与修复
        ──────────────────────────────────────────────────────────
        macOS  dyld: Library not loaded @rpath/libcadet.0.dylib
          → 将 CADET 发行包中的 lib/*.dylib 复制到 ProtePilot/lib/

        Linux  error while loading shared libraries: libcadet.so.0
          → 将 CADET 发行包中的 lib/*.so* 复制到 ProtePilot/lib/
        """
        ok = True
        sys_name = platform.system()
        lib_dir  = self.engine_dir.parent / "lib"

        print("=" * 58)
        print("  cadet-cli 环境诊断")
        print("=" * 58)
        print(f"  操作系统   : {sys_name} ({platform.machine()})")
        print(f"  引擎目录   : {self.engine_dir}")
        print(f"  库目录     : {lib_dir}")
        print()

        # ── 1. 可执行文件 ─────────────────────────────────────────────
        try:
            exe = self._resolve_exe()
            print(f"  [✓] cadet-cli  → {exe}")
        except FileNotFoundError as e:
            print(f"  [✗] cadet-cli 未找到：{e}")
            ok = False
            exe = None

        # ── 2. lib/ 目录 ──────────────────────────────────────────────
        if not lib_dir.exists():
            print(f"  [✗] lib/ 目录不存在：{lib_dir}")
            print(f"       → 请创建该目录并复制 CADET 的库文件")
            ok = False
        else:
            ext = ".dylib" if sys_name == "Darwin" else ".so"
            libs = list(lib_dir.glob(f"*{ext}*"))
            if not libs:
                print(f"  [✗] lib/ 目录存在但无 {ext} 文件")
                print(f"       → 请将 CADET 发行包中的 lib/ 内容复制到 {lib_dir}")
                ok = False
            else:
                for lib in sorted(libs):
                    print(f"  [✓] {lib.name}")

        # ── 3. 试运行（始终直接调用绝对路径，不依赖 conda run）────────
        print()
        if ok and exe:
            test_cmd = [str(exe), "--version"]
            run_env = self._build_env()

            r = subprocess.run(
                test_cmd,
                capture_output=True, text=True, timeout=15, env=run_env,
            )
            if r.returncode == 0:
                ver = (r.stdout + r.stderr).strip().splitlines()[0]
                print(f"  [✓] 试运行成功：{ver}")
            else:
                print(f"  [✗] 试运行失败（退出码 {r.returncode}）")
                print(f"      STDERR: {(r.stdout + r.stderr).strip()[:300]}")
                ok = False

        # ── 修复提示 ──────────────────────────────────────────────────
        if not ok:
            print()
            print("  ── 修复方法 ──────────────────────────────────────────")
            if sys_name == "Darwin":
                print("  macOS: 从 CADET 安装包中复制库文件")
                print("    cp <cadet_pkg>/lib/*.dylib  \\")
                print(f"       {lib_dir}/")
            else:
                print("  Linux: 从 CADET 安装包中复制库文件")
                print("    cp <cadet_pkg>/lib/*.so*  \\")
                print(f"       {lib_dir}/")
            print()
            print("  下载地址：https://github.com/cadet/CADET-Core/releases/latest")

        print("=" * 58)
        return ok

    # ═════════════════════════════════════════════════════════════════════
    # 私有：构建带库路径的进程环境变量
    # ═════════════════════════════════════════════════════════════════════

    def _build_env(self) -> dict:
        """
        复制当前环境变量，并将 ProtePilot/lib/ 注入动态库搜索路径。

        macOS : 设置 DYLD_LIBRARY_PATH
        Linux : 设置 LD_LIBRARY_PATH

        这可解决 cadet-cli 找不到 libcadet.0.dylib 的问题，
        即使 @rpath 配置与目录结构不符也能正常加载。
        """
        env     = os.environ.copy()
        lib_dir = self.engine_dir.parent / "lib"

        if lib_dir.exists():
            sys_name = platform.system()
            if sys_name == "Darwin":
                key = "DYLD_LIBRARY_PATH"
            else:
                key = "LD_LIBRARY_PATH"

            existing = env.get(key, "")
            env[key] = f"{lib_dir}:{existing}" if existing else str(lib_dir)
            log.info("  库路径  : %s=%s", key, env[key])
        else:
            log.warning(
                "lib/ 目录不存在（%s），动态库可能无法加载。\n"
                "  → 请将 CADET 发行包的 lib/ 内容复制到该目录。",
                lib_dir,
            )

        return env

    # ═════════════════════════════════════════════════════════════════════
    # 私有：定位 cadet-cli 可执行文件
    # ═════════════════════════════════════════════════════════════════════

    def _resolve_exe(self) -> Path:
        """
        按优先级查找 cadet-cli:
          1. conda 环境 cadet_env 中的 cadet-cli（自动管理所有依赖库）
          2. engine/cadet-cli.exe   (Windows)
          3. engine/cadet-cli       (Linux / macOS)
          4. 系统 PATH 中的 cadet-cli

        全部找不到时抛出 FileNotFoundError，提示下载链接。
        """
        # ── 优先：conda cadet_env ──────────────────────────────────────────
        conda_candidates = [
            Path("/opt/miniconda3/envs/cadet_env/bin/cadet-cli"),
            Path("/opt/anaconda3/envs/cadet_env/bin/cadet-cli"),
            Path.home() / "miniconda3/envs/cadet_env/bin/cadet-cli",
            Path.home() / "anaconda3/envs/cadet_env/bin/cadet-cli",
        ]
        for p in conda_candidates:
            if p.exists():
                return p  # 标记为 conda 路径，_build_cmd 会识别

        # ── 本地 engine/ 目录 ─────────────────────────────────────────────
        candidates = [
            self.engine_dir / "cadet-cli.exe",
            self.engine_dir / "cadet-cli",
        ]
        for exe in candidates:
            if exe.exists():
                return exe

        found = shutil.which("cadet-cli") or shutil.which("cadet-cli.exe")
        if found:
            return Path(found)

        raise FileNotFoundError(
            "未找到 cadet-cli 可执行文件。\n"
            f"请将 cadet-cli（或 .exe）放入：{self.engine_dir}\n"
            "下载：https://github.com/cadet/CADET-Core/releases/latest"
        )

    def _build_cmd(self, exe: Path, h5_path: Path) -> list:
        """
        Build the subprocess command for cadet-cli execution.

        Always uses the absolute binary path directly (avoids conda run
        failures in CI/test environments where conda activate is unreliable).
        Library paths are injected via _build_env() instead.
        """
        return [str(exe), str(h5_path)]


# ─────────────────────────────────────────────────────────────────────────────
# __main__：可直接运行的演示入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 参数定义 ──────────────────────────────────────────────────────────
    # （实际项目中可由 src/models/ 下的 AI 预测模型提供）
    PROT_PARAMS = ProteinParams(
        ka      = 35.5,     # 吸附速率  [m³/(mol·s)]
        kd      = 1000.0,   # 解吸速率  [1/s]
        lambda_ = 1200.0,   # 离子容量  [mol/m³_s]
        nu      = 4.7,      # 特征电荷  [–]
        sigma   = 11.83,    # 位阻因子  [–]
    )
    PROC_PARAMS = ProcessParams(
        length       = 0.25,       # 柱长    [m]
        col_porosity = 0.37,       # 孔隙率  [–]
        velocity     = 5.75e-4,    # 流速    [m/s]
        flow_rate    = 1.0e-6,     # 流量    [m³/s]
        col_disp     = 5.75e-8,    # 弥散    [m²/s]
        c_load       = 1.0,        # 进样浓度 [mol/m³]
        salt_load    = 50.0,       # 上样盐  [mol/m³]
        salt_elute   = 500.0,      # 洗脱终盐 [mol/m³]
        ncol         = 100,
        nthreads     = 1,        # conda 版 cadet-cli 为单线程版本
    )

    # ── 引擎初始化（路径相对于项目根目录）──────────────────────────────────
    # 若直接在 src/ 下运行，workspace/engine_dir 需要往上一级
    _here = Path(__file__).resolve().parent          # src/
    _root = _here.parent                             # ProtePilot/

    engine = CadetEngine(
        workspace  = _root / "data",
        engine_dir = _root / "engine",
    )

    RUN_NAME = "iex_sma_demo"

    # ── 步骤 1：构建 HDF5 ────────────────────────────────────────────────
    print("\n" + "─" * 56)
    print("  步骤 1 / 3  ─  构建 CADET HDF5 仿真文件")
    print("─" * 56)
    h5_file = engine.build_h5(f"{RUN_NAME}.h5", PROT_PARAMS, PROC_PARAMS)

    # ── 步骤 2：打印 HDF5 结构（调试验证）──────────────────────────────────
    print("\n" + "─" * 56)
    print("  HDF5 关键路径预览")
    print("─" * 56)
    with h5py.File(h5_file, "r") as _f:
        def _tree(name, obj):
            indent = "  " + "  " * name.count("/")
            if isinstance(obj, h5py.Dataset):
                v = obj[()]
                if v.ndim == 0:
                    print(f"{indent}[DS] {name.split('/')[-1]:30s} = {v}")
                else:
                    print(f"{indent}[DS] {name.split('/')[-1]:30s} shape={v.shape} dtype={v.dtype}")
            else:
                print(f"{indent}[G]  {name.split('/')[-1]}/")
        _f.visititems(_tree)

    # ── 步骤 3：引擎诊断（验证 cadet-cli + 动态库）────────────────────
    print("\n" + "─" * 56)
    print("  步骤 2a / 3  ─  引擎环境诊断")
    print("─" * 56)
    if not engine.check_engine():
        print(
            "\n  引擎未就绪，无法执行仿真。\n"
            "  请按上方提示将 CADET lib/ 目录内容复制到 ProtePilot/lib/\n"
            "  后重新运行此脚本。"
        )
        sys.exit(1)

    # ── 步骤 3：运行仿真 ─────────────────────────────────────────────────
    print("\n" + "─" * 56)
    print("  步骤 2b / 3  ─  调用 cadet-cli 执行仿真")
    print("─" * 56)
    try:
        result = engine.run_simulation(
            f"{RUN_NAME}.h5",
            timeout    = 600,
            auto_build = False,   # HDF5 已在步骤 1 构建
        )
        # 导出 CSV
        csv_out = engine.export_csv(result, stem=RUN_NAME)
        print(f"\n  CSV 结果 → {csv_out}")
        # 输出摘要
        print(result.summary())
        # 绘图
        print("\n" + "─" * 56)
        print("  步骤 3 / 3  ─  绘制层析图")
        print("─" * 56)
        png_out = _root / "data" / f"{RUN_NAME}_plot.png"
        engine.plot_chromatogram(
            result,
            save_path = png_out,
            show      = True,
            title     = f"IEX-SMA  |  Protein ka={PROT_PARAMS.ka}  kd={PROT_PARAMS.kd}",
        )
        print(f"  图表已保存 → {png_out}")

    except FileNotFoundError as exc:
        print(f"\n[跳过仿真] {exc}")
        print(
            "\n  HDF5 文件已生成，可在安装 cadet-cli 后再运行：\n"
            f"    python {Path(__file__).name}\n"
            "  或手动调用：\n"
            f"    engine/cadet-cli  {h5_file}"
        )
        sys.exit(0)

    except RuntimeError as exc:
        print(f"\n[仿真错误] {exc}", file=sys.stderr)
        sys.exit(1)
