import time
import numpy as np
import matplotlib.pyplot as plt
from src.cadet_engine import CadetEngine, ProcessParams
from src.PropertyMapper import PropertyMapper, ProteinProperties

def test_m2_integration():
    print("=== ProtePilot M2: 从基因修饰到杂质图谱的全链路联合测试 ===")

    # 1. 业务逻辑输入：定义含有 PTM 的单抗分子
    protein = ProteinProperties(
        name="mAb_PTM_Test",
        pI=8.5,
        MW_kDa=150.0,
        hydrophobicity=0.35,
        ptm_profile={
            "deamidation_sites": 1,  # 1个脱酰胺位点 -> 生成酸性峰
            "oxidation_sites": 1     # 1个氧化位点 -> 生成碱性峰
        }
    )

    # 2. 映射层：PTM -> 物理参数
    print("\n[1/3] 正在运行 PropertyMapper (生物信息 -> 物理化学)...")
    mapper = PropertyMapper()
    # 使用 M2 新增的快捷方法直接获取 VariantParams 对象
    variants = mapper.map_to_variant_params(protein)

    print("\n映射生成的物理参数 (VariantParams):")
    print(f"  Acidic (脱酰胺): nu={variants.acidic.nu:.3f}, ka={variants.acidic.ka:.6f}")
    print(f"  Main   (主峰)  : nu={variants.main.nu:.3f}, ka={variants.main.ka:.6f}")
    print(f"  Basic  (氧化)  : nu={variants.basic.nu:.3f}, ka={variants.basic.ka:.6f}")

    # 3. 物理层：执行 CADET 仿真
    print("\n[2/3] 正在运行 CadetEngine (多组分竞争吸附仿真)...")
    engine = CadetEngine(workspace="data", engine_dir="engine")
    process = ProcessParams()  # 默认梯度 15 mM/min

    h5_path = engine.build_h5("test_m2_integration.h5", variants, process)
    result = engine.run_simulation(h5_path)

    # 4. 结果分析与 CQA 输出
    print("\n[3/3] 仿真完成！正在计算 CQA (分离度)...")
    cqa = result.compute_cqa()

    print("\n--- CQA 质量报告 ---")
    print("保留时间 (RT):")
    for name in ("Acidic", "Main", "Basic"):
        p = cqa["peaks"][name]
        print(f"  {name:7s}: RT = {p['rt_min']:.2f} min,  "
              f"FWHM = {p['fwhm_min']:.3f} min,  "
              f"Height = {p['height']:.6f} mol/m³")

    print("\n分离度 (Resolution):")
    for label, rs in cqa["resolution"].items():
        status = "✅ 基线分离" if rs >= 1.5 else ("⚠️ 部分重叠" if rs >= 0.8 else "❌ 未分离")
        print(f"  {label}: Rs = {rs:.3f}  {status}")

    print("\n面积百分比:")
    for name, pct in cqa["area_pct"].items():
        print(f"  {name}: {pct:.1f}%")

    # 5. 绘制带时间戳的工业级图谱
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_filename = f"m2_integration_{timestamp}.png"

    t_min = result.time / 60.0
    fig, ax1 = plt.subplots(figsize=(12, 6))

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

    # 动态自适应视野：聚焦出峰区域
    peaks = cqa["peaks"]
    rts = [peaks[n]["rt_min"] for n in ("Acidic", "Main", "Basic") if peaks[n]["rt_min"] > 0]
    if rts:
        x_lo = max(0, min(rts) - 5)
        x_hi = max(rts) + 8
        ax1.set_xlim(x_lo, x_hi)
    else:
        ax1.set_xlim(0, 50)

    ax1.set_title(f"M2 Integration: PTMs → Chromatogram ({protein.name})", fontsize=14)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_filename, dpi=300)
    print(f"\n✅ 联合测试成功！层析图已保存为: {plot_filename}")

if __name__ == "__main__":
    test_m2_integration()
