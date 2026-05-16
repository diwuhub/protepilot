import time
import numpy as np
import matplotlib.pyplot as plt
from src.cadet_engine import CadetEngine, VariantParams, ProcessParams

def test_multicomponent():
    print("=== 测试 ProtePilot M1: 多组分电荷变体仿真 ===")

    engine = CadetEngine(workspace="data", engine_dir="engine")

    # ── 物理正确的参数 ──────────────────────────────────────────────
    # ka = 1000 × 0.25^ν  (与 MainOrchestrator 动态公式一致)
    # 酸性变体 (脱酰胺): ν↓ → ka↑ → 弱保留，早出峰
    # 碱性变体 (氧化):   ν↑ → ka↓ → 强保留，晚出峰
    nu_acidic = 4.5
    nu_main   = 5.0
    nu_basic  = 5.5

    variant_dict = {
        "acidic": {
            "ka": 1000 * (0.25 ** nu_acidic),   # ~1.953
            "nu": nu_acidic,
            "sigma": 10.2,   # 脱酰胺轻微膨胀
        },
        "main": {
            "ka": 1000 * (0.25 ** nu_main),      # ~0.977
            "nu": nu_main,
            "sigma": 10.0,
        },
        "basic": {
            "ka": 1000 * (0.25 ** nu_basic),     # ~0.500
            "nu": nu_basic,
            "sigma": 9.8,    # 氧化轻微收缩
        },
    }

    print("\n物理参数:")
    for name in ("acidic", "main", "basic"):
        v = variant_dict[name]
        print(f"  {name:7s}: nu={v['nu']:.1f}  ka={v['ka']:.6f}  sigma={v['sigma']}")

    variants = VariantParams.from_dict(variant_dict)
    process = ProcessParams()   # 默认梯度 15 mM/min

    h5_path = engine.build_h5("test_m1_variants.h5", variants, process)
    result = engine.run_simulation(h5_path)

    print(f"\n仿真完成！耗时 {result.wall_time:.2f}s")
    print(f"  时间点数: {len(result.time)}")
    print(f"  组分: {list(result.outlet.keys())}")

    # 检查峰值
    for name in ("Acidic", "Main", "Basic"):
        conc = result.outlet[name]
        peak_val = float(conc.max())
        peak_idx = int(conc.argmax())
        peak_t_min = result.time[peak_idx] / 60.0
        print(f"  {name:7s}: peak={peak_val:.6f} mol/m³ @ {peak_t_min:.2f} min")

    # ── CQA 分离度分析 ──────────────────────────────────────────────
    cqa = result.compute_cqa()
    print("\n分离度 (Resolution):")
    for label, rs in cqa["resolution"].items():
        print(f"  {label}: Rs = {rs:.3f}")
    print("面积百分比:")
    for name, pct in cqa["area_pct"].items():
        print(f"  {name}: {pct:.1f}%")

    # ── 绘制多组分层析图 ──────────────────────────────────────────────
    t_min = result.time / 60.0
    fig, ax1 = plt.subplots(figsize=(12, 6))
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 蛋白浓度 (左 Y 轴)
    ax1.plot(t_min, result.acidic, label="Acidic (Deamidation)",
             color="#FF4B4B", linestyle="--", linewidth=2)
    ax1.plot(t_min, result.main, label="Main Peak",
             color="#1F77B4", linewidth=2.5)
    ax1.plot(t_min, result.basic, label="Basic (Oxidation)",
             color="#2CA02C", linestyle=":", linewidth=2)

    ax1.set_xlabel("Time (min)", fontsize=12)
    ax1.set_ylabel("Protein Concentration (mol/m³)", fontsize=12)

    # 盐梯度 (右 Y 轴)
    ax2 = ax1.twinx()
    ax2.plot(t_min, result.salt, color="gray", alpha=0.4, linewidth=1.5,
             label="Salt gradient")
    ax2.set_ylabel("Salt (mM)", fontsize=12, color="gray")
    ax2.tick_params(axis='y', labelcolor='gray')

    # 动态 X 轴范围：聚焦出峰区域
    peaks = cqa["peaks"]
    rts = [peaks[n]["rt_min"] for n in ("Acidic", "Main", "Basic") if peaks[n]["rt_min"] > 0]
    if rts:
        x_lo = max(0, min(rts) - 5)
        x_hi = max(rts) + 8
        ax1.set_xlim(x_lo, x_hi)
    else:
        ax1.set_xlim(0, 50)

    ax1.set_title("M1: Multi-Component CEX Simulation (Charge Variants)", fontsize=14)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"m1_variants_test_{timestamp}.png", dpi=300)
    print(f"\n✅ 层析图已保存为 m1_variants_test_{timestamp}.png")


if __name__ == "__main__":
    test_multicomponent()
