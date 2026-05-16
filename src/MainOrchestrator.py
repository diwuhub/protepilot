"""
MainOrchestrator.py  ·  ProtePilot
===========================================================
ProtePilot 核心编排器 — v10 三阶段优化器（单调性对数二分法）

闭环工作流
──────────────────────────────────────────────────────────
    蛋白序列 + 属性预测
          │
          ▼
    PropertyMapper        (pI/MW/疏水性 → nu 初始猜测)
          │
          ▼
    Stage 1:   全局电荷扫描   (硬编码 nu 网格，动态 ka 配对)
          │
          ▼
    Stage 1.5: 局部网格加密   (nu_best ± 1.5, step=0.5)
          │
          ▼
    Stage 2:   动力学精调     (锁定 nu，scipy bounded ka)
          │
          ▼
    best_ever 全局最优       (贯穿全程记录)
          │
          ▼
    汇总报告

v10 · 三阶段优化器
──────────────────────────────────────────────────────────
  Stage 1: 全局电荷扫描 (Global ν Scan) — 动态 ka 配对
      ★ 硬编码宽泛网格: nu_grid = [3.0, 5.0, 8.0, 12.0, 16.0, 22.0]
      ★ 动态 ka: ka_test = kd × (c_target / Λ)^ν = 1000 × 0.25^ν
        （物理含义：假设蛋白在 c_salt=300 mM 时 K_eq≈1，
         即 ka/kd × (Λ/c_salt)^ν = 1 → ka = kd × (c_salt/Λ)^ν
         c_salt/Λ = 300/1200 = 0.25）
      ★ kd = 1000.0（CADET SMA 标准值）
      ★ 运行 6 组 (nu, ka_test) 强制仿真
      ★ 取 abs(Sim_RT - Exp_RT) 最小者 → nu_best_stg1

  Stage 1.5: 局部网格加密 (Local ν Refinement)
      ★ 以 Stage 1 胜出的 nu_best_stg1 为中心
      ★ 在 [nu_best_stg1 - 2.0, nu_best_stg1 + 2.0] 区间，step = 0.5
      ★ 继续使用动态 ka = kd × 0.25^ν
      ★ 选出局部最优 nu_fine → 传递给 Stage 2

  Stage 2: 动力学精调 (Kinetic Fine-Tuning) — 单调性对数二分法
      ★ 锁定 Stage 1.5 最优 nu_fine
      ★ 物理保证: RT(ka) 关于 ka 单调递增 → 根搜索问题
      ★ Phase A: 对数探针（12点，以动态 ka 为中心 ±4 decades）
      ★ Phase B: 流穿过渡区加密（消除 FLOWTHRU→有效 的跳变死角）
      ★ Phase C: 对数空间二分（括号搜索 + 二分收敛）
      ★ 回退保护: Stage 2 失败 → 回退 Stage 1.5/Stage 1 最优
      ★ 不使用 scipy（penalty 9999 破坏 Brent 抛物线插值 → 吸引到 40min 边界）

  极端惩罚函数:
      ★ Sim_RT > 40.0 min → 返回 9999.0（蛋白进入再生段 = 死吸附）
      ★ 无峰/流穿 → 返回 9999.0
      ★ 仿真失败 → 返回 9999.0

  SMA 物理背景:
      K_eq = (ka/kd) × (Λ/c_salt)^ν
      当 ν 增大时，(Λ/c_salt)^ν 指数增长 → ka 必须指数缩小以补偿
      ka_test = kd × (c_target/Λ)^ν 精确抵消此指数增长

时间单位换算
──────────────────────────────────────────────────────────
  CADET 仿真时间 [s]  ÷ 60  →  保留时间 [min]（与 Empower 一致）
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time as _time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── 路径设置 ──────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "utils"))

from cadet_engine    import CadetEngine, ProteinParams, ProcessParams
from PropertyMapper  import PropertyMapper, ProteinProperties, MapperConfig
from utils.EmpowerParser import EmpowerParser

log = logging.getLogger("Orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IterationRecord:
    """记录单次优化迭代的快照"""
    iteration:    int
    ka:           float
    nu:           float
    sim_rt_min:   float
    exp_rt_min:   float
    abs_err_min:  float
    rel_err_pct:  float
    converged:    bool
    phase:        str  = ""
    note:         str  = ""


@dataclass
class OrchestrationResult:
    """编排器的完整输出"""
    protein_name:       str
    converged:          bool
    n_iterations:       int
    final_ka:           float
    final_sim_rt_min:   float
    exp_rt_min:         float
    final_err_pct:      float
    threshold_pct:      float
    iterations:         List[IterationRecord] = field(default_factory=list)
    final_sma_params:   Dict[str, float]      = field(default_factory=dict)
    wall_time_s:        float                 = 0.0
    h5_path:            str                   = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["iterations"] = [asdict(r) for r in self.iterations]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# 峰检测工具
# ─────────────────────────────────────────────────────────────────────────────

def detect_peak_rt(time_s: np.ndarray, conc: np.ndarray) -> float:
    """
    从浓度时序中检测蛋白峰的保留时间。
    在洗脱段（600 s 之后）内取最大浓度点；无峰时回退全局最大。
    """
    global_max = conc.max()
    elute_mask = (time_s >= 600.0)
    threshold  = max(1e-9, 0.01 * global_max)
    if elute_mask.any() and conc[elute_mask].max() > threshold:
        idx_local = np.argmax(conc[elute_mask])
        return float(time_s[elute_mask][idx_local])
    return float(time_s[np.argmax(conc)])


# ─────────────────────────────────────────────────────────────────────────────
# 极端惩罚函数
# ─────────────────────────────────────────────────────────────────────────────

def compute_penalized_error(
    sim_rt_min: float,
    exp_rt_min: float,
    max_allowed_rt_min: float = 40.0,
) -> float:
    """
    带物理护栏的惩罚误差函数。

    规则：
    1. 如果 sim_RT > max_allowed_rt_min (40 min) → 返回 9999.0（死吸附惩罚）
    2. 如果 sim_RT < 8.0 min（流穿区，低于柱死时间 L/v≈7.25 min）→ 返回 9999.0
    3. 否则返回 abs(sim_RT - exp_RT) / exp_RT × 100 (%)
    """
    if exp_rt_min <= 0:
        return 9999.0

    # 死吸附惩罚：蛋白进入再生段或更晚
    if sim_rt_min > max_allowed_rt_min:
        return 9999.0

    # 流穿惩罚（柱死时间 L/v ≈ 7.25 min，低于 8.0 min 即为流穿）
    if sim_rt_min < 8.0:
        return 9999.0

    abs_err = abs(sim_rt_min - exp_rt_min)
    base_err = (abs_err / exp_rt_min) * 100.0
    return base_err


# ─────────────────────────────────────────────────────────────────────────────
# 主编排器
# ─────────────────────────────────────────────────────────────────────────────

class MainOrchestrator:
    """
    ProtePilot 工作流编排器 — v8 三阶段优化器。

    Stage 1:   硬编码 nu 网格 [3,5,8,12,16,22]，动态 ka = kd×(0.25)^ν → 粗选 nu
    Stage 1.5: 局部 nu 加密 [nu_best±1.5, step=0.5]，动态 ka → 精选 nu_fine
    Stage 2:   锁定 nu_fine，scipy.minimize_scalar 精调 ka ∈ [1e-10, 500.0]

    极端惩罚: Sim_RT>40min 或无峰 → 9999.0
    kd = 1000.0（CADET SMA 标准值，严禁覆盖为 1.0）

    Parameters
    ----------
    workspace    : 数据目录
    engine_dir   : cadet-cli 所在目录
    threshold_pct: 收敛阈值（默认 2%）
    max_iter     : Stage 2 scipy 最大迭代次数（默认 30）
    """

    # ── Stage 1: 全局 nu 网格（硬编码，宽泛覆盖）──────────────────────
    # v10: 增加 nu=1.0, 2.0 覆盖弱电荷蛋白（如 Keytruda pI=8.65 在 IEX 中）
    NU_GRID = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 22.0]

    # ── Stage 1 / 1.5: 动态 ka 配对参数 ─────────────────────────────────
    # ka_test = kd × (c_target / Λ)^ν
    # 物理含义：假设蛋白在 c_salt = 300 mM 时 K_eq ≈ 1
    # c_target / Λ = 300 / 1200 = 0.25
    KA_RATIO = 0.25   # c_target / Λ

    # ── Stage 1.5: 局部网格加密参数 ──────────────────────────────────────
    NU_FINE_HALF_SPAN = 2.0   # nu_best ± 2.0（扩大搜索范围）
    NU_FINE_STEP      = 0.5   # 步长 0.5
    NU_MIN            = 1.0   # v10: 降低至 1.0（SMA 允许 ν ≥ 1.0）

    # ── Stage 2: ka 搜索边界 ─────────────────────────────────────────────
    KA_LO = 1e-10
    KA_HI = 500.0

    # ── kd: CADET SMA 标准值（严禁覆盖为 1.0）─────────────────────────
    KD_REFERENCE = 1000.0

    # ── 惩罚值 ────────────────────────────────────────────────────────
    PENALTY_LOSS = 9999.0

    # ── 最大允许 RT（超过即为死吸附）──────────────────────────────────
    MAX_ALLOWED_RT_MIN = 40.0

    def __init__(
        self,
        workspace:     str | Path = "data",
        engine_dir:    str | Path = "engine",
        threshold_pct: float      = 2.0,
        max_iter:      int        = 30,
        **kwargs,
    ):
        self.workspace     = Path(workspace).resolve()
        self.engine_dir    = Path(engine_dir).resolve()
        self.threshold_pct = threshold_pct
        self.max_iter      = max_iter

        self.engine  = CadetEngine(workspace=self.workspace, engine_dir=self.engine_dir)
        self.mapper  = PropertyMapper()

    # ── 公开入口 ──────────────────────────────────────────────────────

    def run(
        self,
        protein:        ProteinProperties,
        process_params: Optional[ProcessParams]  = None,
        exp_txt_path:   Optional[str | Path]     = None,
        exp_component:  str                      = "mAb_Main_Peak",
        save_report:    bool                     = True,
        exp_rt_override: Optional[float]         = None,
    ) -> OrchestrationResult:
        """完整闭环工作流入口。"""
        t0 = _time.monotonic()

        print()
        print("=" * 80)
        print("  ProtePilot — 闭环仿真优化（v8 三阶段优化器）")
        print(f"  蛋白 : {protein.name}")
        print("=" * 80)

        # ── Step 1: PropertyMapper → 初始 nu 参考 ─────────────────────
        print("\n[Step 1/5] PropertyMapper：物理属性 → 初始参数参考")
        sma_dict = self.mapper.map(protein)
        print(self.mapper.explain(protein))

        init_nu      = sma_dict["nu"]
        fixed_lambda = sma_dict["lambda_"]
        fixed_sigma  = sma_dict["sigma"]
        mapper_kd    = sma_dict["kd"]

        # ★ 安全检查: 确保 kd=1000.0，严禁 1.0
        if mapper_kd < 100.0:
            print(f"  ⚠ 警告: PropertyMapper 返回 kd={mapper_kd}，覆盖为 {self.KD_REFERENCE}")
        used_kd = self.KD_REFERENCE

        proc_params = process_params or ProcessParams()

        # 动态计算梯度结束时间
        elute_duration_s = (proc_params.salt_elute - proc_params.salt_load) / proc_params.gradient_slope * 60.0
        t_elute_end_s = self.engine.T_WASH_END + elute_duration_s
        gradient_end_min = t_elute_end_s / 60.0

        print(f"  PropertyMapper nu 参考 = {init_nu:.3f}（仅供参考，Stage 1 使用全局网格 + 动态 ka）")
        print(f"  固定 kd = {used_kd:.1f}（CADET SMA 标准值）")
        print(f"  梯度结束 = {gradient_end_min:.1f} min ({t_elute_end_s:.0f} s)")
        print(f"  死吸附阈值 = {self.MAX_ALLOWED_RT_MIN:.1f} min")

        # ── Step 2: 加载实验参考数据 ───────────────────────────────────
        if exp_rt_override is not None:
            print(f"\n[Step 2/5] 实验 RT（直接传入）: {exp_rt_override:.3f} min")
            exp_rt_min = exp_rt_override
        else:
            print("\n[Step 2/5] EmpowerParser：加载实验参考 RT")
            exp_rt_min = self._load_exp_rt(exp_txt_path, exp_component)
        exp_rt_s = exp_rt_min * 60.0
        print(f"  实验参考 RT：{exp_rt_min:.3f} min ({exp_rt_s:.1f} s)")

        # ── 优化共享状态 ──────────────────────────────────────────────
        records: List[IterationRecord] = []
        iter_counter = [0]

        best_ever = {
            "ka": 1.0, "nu": init_nu,
            "sim_rt_min": 0.0, "rel_err_pct": 9999.0,
            "abs_err": 9999.0, "h5_path": "",
        }

        # ── 单次仿真函数（纯仿真，无任何缩放）────────────────────────
        def run_sim(ka_val: float, nu_val: float, phase: str) -> Tuple[float, float]:
            """
            运行一次真实 CADET 仿真。
            返回 (sim_rt_min, penalized_err)。
            仿真失败/流穿/死吸附 → (0.0, 9999.0)。
            """
            iter_counter[0] += 1
            i = iter_counter[0]

            prot_params = ProteinParams(
                ka=ka_val, kd=used_kd,
                lambda_=fixed_lambda, nu=nu_val, sigma=fixed_sigma,
            )

            safe_name = protein.name.replace(" ", "_")
            h5_name = f"opt_{phase}_{i:03d}_{safe_name}.h5"
            h5_path = self.workspace / h5_name
            try:
                if h5_path.exists():
                    h5_path.unlink()
            except OSError:
                h5_name = f"opt_{phase}_{i:03d}_{safe_name}_{int(_time.time())}.h5"
                h5_path = self.workspace / h5_name

            try:
                h5_built = self.engine.build_h5(h5_name, prot_params, proc_params)
                sim_result = self.engine.run_simulation(
                    h5_built, timeout=300, auto_build=False,
                )
            except (RuntimeError, FileNotFoundError) as e:
                log.warning("  仿真失败 (ka=%.3f, nu=%.3f): %s",
                            ka_val, nu_val, str(e)[:120])
                rec = IterationRecord(
                    iteration=i, ka=round(ka_val, 4), nu=round(nu_val, 4),
                    sim_rt_min=0.0, exp_rt_min=round(exp_rt_min, 3),
                    abs_err_min=9999.0, rel_err_pct=9999.0,
                    converged=False, phase=phase, note="SIM_FAIL",
                )
                records.append(rec)
                self._print_iter(rec)
                return 0.0, self.PENALTY_LOSS

            # 峰检测
            protein_conc = sim_result.outlet.get(
                "Protein",
                sim_result.outlet.get(
                    list(sim_result.outlet.keys())[-1], np.array([0.0])
                ),
            )
            sim_rt_s   = detect_peak_rt(sim_result.time, protein_conc)
            sim_rt_min = sim_rt_s / 60.0

            # 流穿检测（Elute 从 600 s = 10 min 开始）
            elute_start_min = self.engine.T_WASH_END / 60.0
            if sim_rt_min < elute_start_min:
                rec = IterationRecord(
                    iteration=i, ka=round(ka_val, 4), nu=round(nu_val, 4),
                    sim_rt_min=round(sim_rt_min, 3), exp_rt_min=round(exp_rt_min, 3),
                    abs_err_min=round(abs(sim_rt_min - exp_rt_min), 3),
                    rel_err_pct=9999.0,
                    converged=False, phase=phase, note="FLOWTHRU",
                )
                records.append(rec)
                self._print_iter(rec)
                return sim_rt_min, self.PENALTY_LOSS

            # 极端惩罚: 死吸附检测
            penalized_err = compute_penalized_error(
                sim_rt_min, exp_rt_min, self.MAX_ALLOWED_RT_MIN,
            )

            # 真实误差（用于收敛判定和 best_ever）
            abs_err = abs(sim_rt_min - exp_rt_min)
            real_err = (abs_err / exp_rt_min) * 100.0 if exp_rt_min > 0 else 0.0
            converged = real_err <= self.threshold_pct

            # 更新 best_ever（仅当有效洗脱时）
            if real_err < best_ever["rel_err_pct"] and sim_rt_min <= self.MAX_ALLOWED_RT_MIN:
                best_ever["ka"]          = ka_val
                best_ever["nu"]          = nu_val
                best_ever["sim_rt_min"]  = sim_rt_min
                best_ever["rel_err_pct"] = real_err
                best_ever["abs_err"]     = abs_err
                best_ever["h5_path"]     = str(h5_path)

            # 标注
            note_str = "CONVERGE" if converged else phase
            if sim_rt_min > self.MAX_ALLOWED_RT_MIN:
                note_str = "DEAD_ADS!"  # 死吸附警告
            if penalized_err >= self.PENALTY_LOSS:
                note_str = "PENALTY!"

            rec = IterationRecord(
                iteration=i, ka=round(ka_val, 4), nu=round(nu_val, 4),
                sim_rt_min=round(sim_rt_min, 3), exp_rt_min=round(exp_rt_min, 3),
                abs_err_min=round(abs_err, 3), rel_err_pct=round(real_err, 2),
                converged=converged, phase=phase, note=note_str,
            )
            records.append(rec)
            self._print_iter(rec)

            return sim_rt_min, penalized_err

        # ==================================================================
        #  Stage 1: 全局电荷扫描 (Global ν Scan)
        #  硬编码网格 [3, 5, 8, 12, 16, 22]
        #  动态 ka: ka_test = kd × (0.25)^ν — 抵消 ν 的指数增长
        # ==================================================================

        # 预计算每个 nu 对应的动态 ka_test
        nu_ka_pairs = []
        for nu_try in self.NU_GRID:
            ka_test = used_kd * (self.KA_RATIO ** nu_try)
            nu_ka_pairs.append((nu_try, ka_test))

        print(f"\n[Step 3/5] Stage 1: 全局电荷扫描（动态 ka 配对）")

        self._print_stage_header(
            "Stage 1", "全局电荷扫描 ν（动态 ka = kd × 0.25^ν）",
            [f"kd = {used_kd:.1f}（CADET SMA 标准值）",
             f"ka_test = kd × (c_target/Λ)^ν = {used_kd:.0f} × 0.25^ν",
             f"nu 网格及配对 ka:"]
            + [f"  nu={nu:.1f} → ka={ka:.6e}" for nu, ka in nu_ka_pairs]
            + [f"死吸附阈值 = Sim_RT > {self.MAX_ALLOWED_RT_MIN} min → 惩罚 9999",
               f"目标: 锁定 abs(Sim_RT - Exp_RT) 最小的 (nu, ka) 组合"],
        )

        stage1_results = []  # [(nu, ka_test, sim_rt_min, abs_err, effective_err)]

        for nu_try, ka_test in nu_ka_pairs:
            sim_rt, pen_err = run_sim(ka_test, nu_try, phase="STG1")
            abs_err = abs(sim_rt - exp_rt_min) if sim_rt > 0 else 9999.0
            effective_err = abs_err if pen_err < self.PENALTY_LOSS else 9999.0
            stage1_results.append((nu_try, ka_test, sim_rt, abs_err, effective_err))

        # 打印 Stage 1 完整结果表
        print()
        print(f"  ┌─ Stage 1 全局扫描结果 ─────────────────────────────────────────────────────")
        print(f"  │  ka_test = {used_kd:.0f} × 0.25^ν（抵消 ν 指数增长）")
        print(f"  │")
        print(f"  │  {'#':>3}  {'nu':>6}  {'ka_test':>12}  {'Sim_RT':>9}  {'Exp_RT':>9}  {'|err|':>9}  {'状态':>10}")
        print(f"  │  " + "-" * 70)

        # 按 effective_err 排序找最优
        sorted_results = sorted(stage1_results, key=lambda x: x[4])
        nu_best_stg1      = sorted_results[0][0]
        ka_best_stg1      = sorted_results[0][1]
        rt_best_stg1      = sorted_results[0][2]

        for rank, (nu_v, ka_v, rt_v, abs_e, eff_e) in enumerate(stage1_results, 1):
            marker = " ★" if nu_v == nu_best_stg1 else "  "
            if eff_e >= 9999.0:
                status = "DEAD/FAIL"
                rt_str = f"{rt_v:>7.2f}m" if rt_v > 0 else " NO PEAK"
                err_str = "  9999.0"
            else:
                status = "OK"
                rt_str = f"{rt_v:>7.2f}m"
                err_str = f"{abs_e:>7.2f}m"
            print(f"  │{marker} {rank:>2}  nu={nu_v:>5.1f}  ka={ka_v:>12.6e}  {rt_str}  "
                  f"{exp_rt_min:>7.2f}m  {err_str}  {status:>10}")

        print(f"  │")
        print(f"  │  ★ Stage 1 粗选 nu = {nu_best_stg1:.1f}  (配对 ka_test = {ka_best_stg1:.6e})")
        if rt_best_stg1 > 0 and sorted_results[0][4] < 9999.0:
            print(f"  │    对应 Sim_RT = {rt_best_stg1:.2f} min")
        else:
            print(f"  │    ⚠ 注意: 最优 nu 点的 Sim_RT 异常，Stage 1.5/2 将尝试修正")
        print(f"  │  → 进入 Stage 1.5 局部加密")
        print(f"  └──────────────────────────────────────────────────────────────────────────")

        # 初始化: 全局 nu 结果列表和锁定 nu（供 Stage 2/2b 使用）
        all_nu_results = list(stage1_results)
        locked_nu      = nu_best_stg1
        locked_ka_test = ka_best_stg1
        locked_nu_rt   = rt_best_stg1

        # 如果 Stage 1 已经收敛，跳过后续
        if best_ever["rel_err_pct"] <= self.threshold_pct:
            print(f"\n  ★ Stage 1 已收敛! 跳过 Stage 1.5 和 Stage 2")
            print(f"    nu={best_ever['nu']:.1f}  ka={best_ever['ka']:.4f}  "
                  f"err={best_ever['rel_err_pct']:.2f}%")

        # ==================================================================
        #  Stage 1.5: 局部网格加密 (Local ν Refinement)
        #  以 Stage 1 胜出的 nu_best_stg1 为中心
        #  在 [nu_best_stg1 - 1.5, nu_best_stg1 + 1.5] 区间, step=0.5
        #  继续使用动态 ka = kd × 0.25^ν
        # ==================================================================
        elif best_ever["rel_err_pct"] > self.threshold_pct:
            nu_fine_lo = max(nu_best_stg1 - self.NU_FINE_HALF_SPAN, self.NU_MIN)
            nu_fine_hi = nu_best_stg1 + self.NU_FINE_HALF_SPAN
            # 生成局部网格（含端点），排除已在 Stage 1 中测试过的 nu_best_stg1
            nu_fine_grid_raw = np.arange(nu_fine_lo, nu_fine_hi + 0.01, self.NU_FINE_STEP)
            nu_fine_grid = [nv for nv in nu_fine_grid_raw
                           if abs(nv - nu_best_stg1) > 0.01]  # 跳过已测过的中心点

            # 如果局部网格为空（极罕见），跳过 Stage 1.5
            if len(nu_fine_grid) == 0:
                nu_fine_grid = [nu_best_stg1]  # fallback

            # 预计算局部网格的动态 ka
            nu_fine_ka_pairs = []
            for nv in nu_fine_grid:
                ka_v = used_kd * (self.KA_RATIO ** nv)
                nu_fine_ka_pairs.append((nv, ka_v))

            print(f"\n[Step 3.5/5] Stage 1.5: 局部网格加密（nu_best={nu_best_stg1:.1f} ± {self.NU_FINE_HALF_SPAN}）")

            self._print_stage_header(
                "Stage 1.5", f"局部网格加密 ν ∈ [{nu_fine_lo:.1f}, {nu_fine_hi:.1f}], step={self.NU_FINE_STEP}",
                [f"Stage 1 粗选 nu = {nu_best_stg1:.1f}",
                 f"加密范围: [{nu_fine_lo:.1f}, {nu_fine_hi:.1f}]",
                 f"步长: {self.NU_FINE_STEP}",
                 f"动态 ka: ka_test = {used_kd:.0f} × 0.25^ν",
                 f"局部网格:"]
                + [f"  nu={nv:.1f} → ka={ka:.6e}" for nv, ka in nu_fine_ka_pairs]
                + [f"目标: 在 Stage 1 粗选附近找到更精确的 nu_fine"],
            )

            stg15_results = []
            for nv, ka_v in nu_fine_ka_pairs:
                if best_ever["rel_err_pct"] <= self.threshold_pct:
                    break
                sim_rt_f, pen_err_f = run_sim(ka_v, nv, phase="STG1.5")
                abs_err_f = abs(sim_rt_f - exp_rt_min) if sim_rt_f > 0 else 9999.0
                eff_err_f = abs_err_f if pen_err_f < self.PENALTY_LOSS else 9999.0
                stg15_results.append((nv, ka_v, sim_rt_f, abs_err_f, eff_err_f))

            # 合并 Stage 1 + Stage 1.5 结果，选全局最优 nu
            all_nu_results = stage1_results + stg15_results
            all_sorted = sorted(all_nu_results, key=lambda x: x[4])
            locked_nu      = all_sorted[0][0]
            locked_ka_test = all_sorted[0][1]
            locked_nu_rt   = all_sorted[0][2]

            # 打印 Stage 1.5 结果
            print()
            print(f"  ┌─ Stage 1.5 局部加密结果 ───────────────────────────────────────────────")
            print(f"  │  加密范围: [{nu_fine_lo:.1f}, {nu_fine_hi:.1f}], step={self.NU_FINE_STEP}")
            print(f"  │")
            for nv, ka_v, rt_v, abs_e, eff_e in stg15_results:
                marker = " ★" if abs(nv - locked_nu) < 0.01 else "  "
                if eff_e >= 9999.0:
                    status_s = "DEAD/FAIL"
                    rt_str = f"{rt_v:>7.2f}m" if rt_v > 0 else " NO PEAK"
                else:
                    status_s = "OK"
                    rt_str = f"{rt_v:>7.2f}m"
                print(f"  │{marker}  nu={nv:>5.1f}  ka={ka_v:>12.6e}  {rt_str}  "
                      f"{exp_rt_min:>7.2f}m  {abs_e:>7.2f}m  {status_s:>10}")
            print(f"  │")
            print(f"  │  ★ 最终锁定 nu = {locked_nu:.1f}  (来源: {'Stage 1' if locked_nu in [r[0] for r in stage1_results] else 'Stage 1.5'})")
            if locked_nu_rt > 0:
                print(f"  │    对应 Sim_RT = {locked_nu_rt:.2f} min  |  配对 ka_test = {locked_ka_test:.6e}")
            print(f"  │  后续 Stage 2 中 nu 将完全固定，绝不再改动")
            print(f"  └──────────────────────────────────────────────────────────────────────────")

        # ==================================================================
        #  Stage 2: 单调性对数二分法 (Log-Space Monotone Bisection)
        #
        #  物理依据:
        #    对于固定的 ν, kd, Λ, 梯度:
        #    ka ↑ → K_eq ↑ → 保留增强 → RT ↑
        #    即 RT(ka) 关于 ka 是单调递增函数
        #
        #  因此: 找 ka* 使得 RT(ka*) = exp_RT 是一维单调根搜索
        #
        #  为什么不用 scipy minimize_scalar:
        #    penalty 值 (9999) 破坏 Brent 抛物线插值 →
        #    优化器被吸引到 penalty 边界 (RT≈40 min) →
        #    所有蛋白得到相同 Sim_RT → 0/6 收敛
        #
        #  算法:
        #    A. 对数探针 (12点): 以 ka_stg15 为中心 ±4 decades
        #    B. 流穿过渡区加密: 精化 FLOWTHRU→有效 的跳变
        #    C. 括号搜索 + 对数二分: 在有效括号内收敛
        #
        #  回退保护: Stage 2 失败 → 回退到 Stage 1/1.5 最优
        # ==================================================================
        if best_ever["rel_err_pct"] > self.threshold_pct:
            print(f"\n[Step 4/5] Stage 2: 单调性对数二分法（锁定 nu={locked_nu:.1f}）")

            # 回退保护: 保存 Stage 1 / 1.5 最优状态
            pre_stg2_backup = dict(best_ever)

            # 以 Stage 1.5 动态 ka 为中心
            ka_center = used_kd * (self.KA_RATIO ** locked_nu)
            log_ka_center = math.log10(max(ka_center, 1e-15))
            # 探针范围: 中心 ±4 decades，钳位到全局边界
            log_lo = max(math.log10(self.KA_LO), log_ka_center - 4)
            log_hi = min(math.log10(self.KA_HI), log_ka_center + 4)

            # 评估预算分配
            budget = self.max_iter
            N_PROBES = min(12, max(5, budget - 8))  # 留至少 8 次给二分

            self._print_stage_header(
                "Stage 2", f"单调性对数二分法（锁定 nu={locked_nu:.1f}）",
                [f"nu = {locked_nu:.1f} (锁定，不可更改)",
                 f"kd = {used_kd:.1f} (CADET SMA 标准值)",
                 f"ka_center = {ka_center:.6e} (动态公式)",
                 f"探针范围 = [10^{log_lo:.1f}, 10^{log_hi:.1f}] ({N_PROBES} 个探针)",
                 f"物理原理: RT(ka) 单调递增 → 根搜索",
                 f"死吸附阈值 = Sim_RT > {self.MAX_ALLOWED_RT_MIN} min",
                 f"目标 < {self.threshold_pct}%",
                 f"回退保护: Stage 2 失败 → 回退 Stage 1/1.5 最优"],
            )

            # ── 调用二分法辅助方法 ─────────────────────────────────────
            evals_used = self._bisect_ka_for_nu(
                run_sim, locked_nu, exp_rt_min, used_kd,
                best_ever, budget, phase="STG2",
            )

            # ── 回退保护检查 ─────────────────────────────────────────────
            stg2_improved = (best_ever["rel_err_pct"] < pre_stg2_backup["rel_err_pct"])

            if not stg2_improved:
                best_ever.update(pre_stg2_backup)

            print()
            print(f"  ┌─ Stage 2 结果 ───────────────────────────────────────────")
            print(f"  │  对数二分法 评估次数: {evals_used}")
            print(f"  │  best_ever ka  = {best_ever['ka']:.6e}")
            print(f"  │  best_ever nu  = {best_ever['nu']:.1f}")
            print(f"  │  best_ever RT  = {best_ever['sim_rt_min']:.3f} min")
            print(f"  │  best_ever err = {best_ever['rel_err_pct']:.2f}%")
            if best_ever["rel_err_pct"] > self.threshold_pct:
                print(f"  │  ⚠ 未收敛 → 将尝试备选 nu（Stage 2b）")
            else:
                print(f"  │  ✓ 收敛! 误差 {best_ever['rel_err_pct']:.2f}% ≤ {self.threshold_pct}%")
            print(f"  └──────────────────────────────────────────────────────────")

        # ==================================================================
        #  Stage 2b: 多-ν 重试 (Multi-ν Fallback)
        #
        #  物理原理:
        #    不同 ν 的流穿边界不同 → 最低可达 RT 不同。
        #    Stage 1/1.5 通过动态 ka 粗筛选 ν，但动态 ka 不代表最优 ka。
        #    某个 ν 在粗筛时排名第二，但在精调 ka 后可能比第一更好。
        #
        #  算法: 取 Stage 1+1.5 中 effective_err 排名 2~4 的 ν，
        #        逐一运行紧凑二分法 (budget=20)，首个收敛即停止。
        # ==================================================================
        if best_ever["rel_err_pct"] > self.threshold_pct:
            # 收集备选 nu（排除已在 Stage 2 中用过的 locked_nu）
            retry_candidates = [
                (nu_v, ka_v, eff_e)
                for nu_v, ka_v, rt_v, abs_e, eff_e in all_nu_results
                if abs(nu_v - locked_nu) > 0.3 and eff_e < 9999.0
            ]
            # v10: 按 effective_err 排序（修复排序 bug），去重
            seen_nu = set()
            unique_retry = []
            for nu_v, ka_v, eff_e in sorted(retry_candidates,
                                             key=lambda x: x[2]):
                nu_r = round(nu_v, 1)
                if nu_r not in seen_nu:
                    seen_nu.add(nu_r)
                    unique_retry.append((nu_v, ka_v))

            if unique_retry:
                print(f"\n[Step 4b/5] Stage 2b: 多-ν 重试")
                print(f"  locked_nu={locked_nu:.1f} 下最佳误差 "
                      f"{best_ever['rel_err_pct']:.2f}% > {self.threshold_pct}%")
                print(f"  备选 nu: {[f'{c[0]:.1f}' for c in unique_retry[:3]]}")

                for alt_nu, alt_ka in unique_retry[:3]:
                    if best_ever["rel_err_pct"] <= self.threshold_pct:
                        break

                    print(f"\n  ── 尝试 nu={alt_nu:.1f} ──")
                    alt_evals = self._bisect_ka_for_nu(
                        run_sim, alt_nu, exp_rt_min, used_kd,
                        best_ever, budget=20, phase="STG2b",
                    )
                    print(f"  ── nu={alt_nu:.1f}: {alt_evals} evals, "
                          f"best_err={best_ever['rel_err_pct']:.2f}% ──")

                print()
                print(f"  ┌─ Stage 2b 最终结果 ──────────────────────────────────────")
                print(f"  │  best_ever ka  = {best_ever['ka']:.6e}")
                print(f"  │  best_ever nu  = {best_ever['nu']:.1f}")
                print(f"  │  best_ever RT  = {best_ever['sim_rt_min']:.3f} min")
                print(f"  │  best_ever err = {best_ever['rel_err_pct']:.2f}%")
                if best_ever["rel_err_pct"] <= self.threshold_pct:
                    print(f"  │  ✓ 多-ν 重试成功!")
                else:
                    print(f"  │  ⚠ 多-ν 重试后仍未收敛")
                print(f"  └──────────────────────────────────────────────────────────")

        # ── 提取最终结果（始终使用 best_ever）──────────────────────────
        final_ka         = best_ever["ka"]
        final_nu         = best_ever["nu"]
        final_sim_rt_min = best_ever["sim_rt_min"]
        final_err_pct    = best_ever["rel_err_pct"]
        converged        = final_err_pct <= self.threshold_pct

        # ── Step 5: 汇总报告 ───────────────────────────────────────────
        print(f"\n[Step 5/5] 生成报告")
        wall_time = _time.monotonic() - t0

        orch_result = OrchestrationResult(
            protein_name       = protein.name,
            converged          = converged,
            n_iterations       = len(records),
            final_ka           = final_ka,
            final_sim_rt_min   = final_sim_rt_min,
            exp_rt_min         = exp_rt_min,
            final_err_pct      = final_err_pct,
            threshold_pct      = self.threshold_pct,
            iterations         = records,
            final_sma_params   = {
                "ka":      round(final_ka,            4),
                "kd":      round(used_kd,             4),
                "lambda_": round(fixed_lambda,        1),
                "nu":      round(final_nu,            4),
                "sigma":   round(fixed_sigma,         4),
            },
            wall_time_s = round(wall_time, 2),
            h5_path     = best_ever["h5_path"],
        )

        summary = self._build_summary(orch_result)
        print(summary)

        if save_report:
            report_path = self.workspace / f"report_{protein.name.replace(' ', '_')}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(orch_result.to_dict(), f, indent=2, ensure_ascii=False)
            log.info("报告已保存 → %s", report_path.name)

        return orch_result

    # ── 私有方法 ──────────────────────────────────────────────────────────

    def _bisect_ka_for_nu(
        self,
        run_sim,
        nu_val: float,
        exp_rt_min: float,
        used_kd: float,
        best_ever: dict,
        budget: int,
        phase: str = "STG2",
    ) -> int:
        """
        对数空间二分法：为给定的 nu 值搜索最优 ka。

        利用 RT(ka) 关于 ka 单调递增的物理性质，
        通过对数探针 + 流穿加密 + 二分收敛找到 ka* 使 RT(ka*) ≈ target。

        Parameters
        ----------
        run_sim     : 仿真闭包，自动更新 best_ever
        nu_val      : 固定的 nu 值
        exp_rt_min  : 实验目标 RT [min]
        used_kd     : kd 参考值
        best_ever   : 全局最优字典（由 run_sim 更新）
        budget      : 本次可用评估次数
        phase       : 阶段标签 (STG2 / STG2b)

        Returns
        -------
        int : 实际使用的评估次数
        """
        # 以动态 ka 为中心
        ka_center = used_kd * (self.KA_RATIO ** nu_val)
        log_ka_ctr = math.log10(max(ka_center, 1e-15))
        log_lo = max(math.log10(self.KA_LO), log_ka_ctr - 4)
        log_hi = min(math.log10(self.KA_HI), log_ka_ctr + 4)

        # 预算分配
        N_PROBES = min(12, max(5, budget - 8))
        evals = 0

        elute_start_min = self.engine.T_WASH_END / 60.0

        # ── Phase A: 对数探针 ────────────────────────────────────────
        probe_kas = np.logspace(log_lo, log_hi, N_PROBES)
        probe_data = []   # [(ka, sim_rt, is_valid, is_flowthru)]
        valid_pairs = []  # [(ka, rt)]

        for ka_p in probe_kas:
            if best_ever["rel_err_pct"] <= self.threshold_pct:
                break
            sim_rt, pen_err = run_sim(float(ka_p), nu_val, phase=phase)
            evals += 1
            is_valid = (pen_err < self.PENALTY_LOSS
                        and sim_rt >= 8.0
                        and sim_rt <= self.MAX_ALLOWED_RT_MIN)
            is_ft = (sim_rt < elute_start_min or pen_err >= self.PENALTY_LOSS)
            probe_data.append((float(ka_p), sim_rt, is_valid, is_ft))
            if is_valid:
                valid_pairs.append((float(ka_p), sim_rt))

        # ── Phase B: 流穿过渡区加密 ─────────────────────────────────
        if (valid_pairs
                and best_ever["rel_err_pct"] > self.threshold_pct):
            valid_sorted = sorted(valid_pairs, key=lambda x: x[0])
            ka_first_valid = valid_sorted[0][0]

            ft_below = [ka for ka, _, _, ft in probe_data
                        if ft and ka < ka_first_valid]
            if ft_below:
                ka_last_ft = max(ft_below)
                log_gap = math.log10(ka_first_valid) - math.log10(ka_last_ft)
                if log_gap > 0.3:
                    n_ref = min(10, budget - evals - 5)
                    if n_ref > 0:
                        ref_kas = np.logspace(
                            math.log10(ka_last_ft),
                            math.log10(ka_first_valid),
                            n_ref + 2,
                        )[1:-1]
                        for ka_r in ref_kas:
                            if best_ever["rel_err_pct"] <= self.threshold_pct:
                                break
                            sim_rt, pen_err = run_sim(float(ka_r), nu_val, phase=phase)
                            evals += 1
                            is_valid = (pen_err < self.PENALTY_LOSS
                                        and sim_rt >= 8.0
                                        and sim_rt <= self.MAX_ALLOWED_RT_MIN)
                            if is_valid:
                                valid_pairs.append((float(ka_r), sim_rt))

        # ── Phase C: 括号搜索 + 对数二分 ────────────────────────────
        if (valid_pairs
                and best_ever["rel_err_pct"] > self.threshold_pct):
            valid_pairs.sort(key=lambda x: x[0])

            ka_lo_bkt = None
            ka_hi_bkt = None
            for ka_v, rt_v in valid_pairs:
                if rt_v <= exp_rt_min:
                    ka_lo_bkt = ka_v
                if rt_v >= exp_rt_min and ka_hi_bkt is None:
                    ka_hi_bkt = ka_v

            if ka_lo_bkt is None and ka_hi_bkt is not None:
                ka_lo_bkt = max(self.KA_LO, valid_pairs[0][0] * 0.01)
            elif ka_hi_bkt is None and ka_lo_bkt is not None:
                ka_hi_bkt = min(self.KA_HI, valid_pairs[-1][0] * 100)
            elif ka_lo_bkt is None and ka_hi_bkt is None:
                ka_lo_bkt = self.KA_LO
                ka_hi_bkt = self.KA_HI

            ka_lo_bkt = max(ka_lo_bkt, self.KA_LO)
            ka_hi_bkt = min(ka_hi_bkt, self.KA_HI)

            while evals < budget:
                if best_ever["rel_err_pct"] <= self.threshold_pct:
                    break
                log_w = abs(math.log10(max(ka_hi_bkt, 1e-15))
                          - math.log10(max(ka_lo_bkt, 1e-15)))
                if log_w < 1e-4:
                    break

                log_m = (math.log10(max(ka_lo_bkt, 1e-15))
                       + math.log10(max(ka_hi_bkt, 1e-15))) / 2.0
                ka_mid = 10 ** log_m

                sim_rt, pen_err = run_sim(float(ka_mid), nu_val, phase=phase)
                evals += 1

                if pen_err >= self.PENALTY_LOSS or sim_rt < 8.0:
                    if ka_mid < ka_center:
                        ka_lo_bkt = ka_mid
                    else:
                        ka_hi_bkt = ka_mid
                elif sim_rt < exp_rt_min:
                    ka_lo_bkt = ka_mid
                else:
                    ka_hi_bkt = ka_mid

        return evals

    def _load_exp_rt(
        self,
        txt_path:  Optional[str | Path],
        component: str,
    ) -> float:
        """加载 Empower 实验 RT；找不到时使用 mock 值"""
        if txt_path is None:
            txt_path = self.workspace / "empower_peak_results.txt"
        txt_path = Path(txt_path)
        if not txt_path.exists():
            log.warning("Empower 文件不存在，使用 mock RT = 15.342 min")
            return 15.342
        parser  = EmpowerParser(txt_path)
        samples = parser.parse_file()
        if not samples:
            log.warning("Empower 文件无数据，使用 mock RT = 15.342 min")
            return 15.342
        sample = samples[0]
        peak   = sample.get_peak(component)
        if peak is None:
            peak = sample.main_peak()
            log.warning("未找到组分 '%s'，使用主峰 '%s'",
                        component, peak.component if peak else "None")
        return peak.rt_min if peak else 15.342

    @staticmethod
    def _print_stage_header(stage_id: str, title: str, details: List[str]):
        """打印阶段头部信息"""
        print()
        print("=" * 80)
        print(f"  {stage_id}: {title}")
        for d in details:
            print(f"    {d}")
        print("=" * 80)
        print(f"  {'Iter':>4}  {'ka':>9}  {'nu':>7}  {'Sim_RT':>9}  "
              f"{'Exp_RT':>9}  {'Error%':>8}  {'phase':>6}  note")
        print("-" * 80)

    @staticmethod
    def _print_iter(r: IterationRecord):
        """★ 强制打印每一轮: [Iter X] nu=..., ka=..., Sim_RT=..., Exp_RT=..., Error=%"""
        marker = "★" if r.converged else " "
        print(f" {marker}[Iter {r.iteration:>3}]  "
              f"nu={r.nu:>7.3f}  ka={r.ka:>9.4f}  "
              f"Sim_RT={r.sim_rt_min:>8.3f}  Exp_RT={r.exp_rt_min:>8.3f}  "
              f"Error={r.rel_err_pct:>7.2f}%  "
              f"{r.phase:>6}  {r.note}")

    @staticmethod
    def _build_summary(r: OrchestrationResult) -> str:
        status_icon = "CONVERGED" if r.converged else "NOT CONVERGED"
        stg1_count  = sum(1 for rec in r.iterations if rec.phase == "STG1")
        stg15_count = sum(1 for rec in r.iterations if rec.phase == "STG1.5")
        stg2_count  = sum(1 for rec in r.iterations if rec.phase == "STG2")
        stg2b_count = sum(1 for rec in r.iterations if rec.phase == "STG2b")
        penalty_count = sum(1 for rec in r.iterations
                           if rec.note in ("PENALTY!", "DEAD_ADS!", "FLOWTHRU"))
        lines = [
            "",
            "=" * 62,
            "  ProtePilot — v9 三阶段优化器（对数二分法）总结报告",
            "=" * 62,
            f"  蛋白          : {r.protein_name}",
            f"  总体结论      : {status_icon}",
            f"  仿真评估次数  : {r.n_iterations}"
            f" (Stage1={stg1_count}, Stage1.5={stg15_count}, Stage2={stg2_count}"
            f"{f', Stage2b={stg2b_count}' if stg2b_count > 0 else ''})",
            f"  惩罚/异常次数 : {penalty_count}",
            f"  收敛阈值      : ±{r.threshold_pct:.1f}%",
            f"  最终误差      : {r.final_err_pct:.2f}%",
            f"  总耗时        : {r.wall_time_s:.1f} s",
            "-" * 62,
            "  最终校准 SMA 参数:",
            f"    ka      = {r.final_sma_params.get('ka', 0):.4f}  m³/(mol·s)",
            f"    kd      = {r.final_sma_params.get('kd', 0):.4f}  1/s",
            f"    nu      = {r.final_sma_params.get('nu', 0):.4f}  (Stage1/1.5 锁定)",
            f"    sigma   = {r.final_sma_params.get('sigma', 0):.4f}",
            f"    lambda  = {r.final_sma_params.get('lambda_', 0):.0f}  mol/m³",
            "-" * 62,
            "  峰保留时间对比:",
            f"    仿真 RT = {r.final_sim_rt_min:.3f} min",
            f"    实验 RT = {r.exp_rt_min:.3f} min",
            f"    误差    = {r.final_err_pct:.2f}%",
            "=" * 62,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# __main__（快速测试）
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    root = _ROOT_DIR

    protein = ProteinProperties(
        name           = "mAb_ExampleA",
        pI             = 8.5,
        MW_kDa         = 148.0,
        hydrophobicity = 0.35,
        pH_working     = 6.0,
    )

    orch = MainOrchestrator(
        workspace     = root / "data",
        engine_dir    = root / "engine",
        threshold_pct = 2.0,
        max_iter      = 30,
    )

    result = orch.run(
        protein       = protein,
        exp_txt_path  = root / "data" / "empower_peak_results.txt",
        exp_component = "mAb_Main_Peak",
    )
