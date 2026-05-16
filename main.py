"""
main.py  ·  ProtePilot
===========================================================
一键运行入口：从蛋白序列到参数自动校准

使用方法
────────────────────────────────────────────────────────────
    # 默认运行（使用内置 mock 蛋白属性）
    python main.py

    # 自定义蛋白属性
    python main.py --name "mAb_ProductX" --pI 8.2 --MW 148 --hydro 0.42 --pH 6.0

    # 指定实验数据文件
    python main.py --exp data/empower_peak_results.txt --component "mAb_Main_Peak"

    # 调整优化参数（Secant + nu tune，10次迭代通常足够）
    python main.py --threshold 2.0 --max-iter 10

命令行参数
────────────────────────────────────────────────────────────
    --name        蛋白名称                  (默认: mAb_ExampleA)
    --pI          等电点 pH                 (默认: 8.5)
    --MW          分子量 kDa                (默认: 148.0)
    --hydro       疏水性得分 [0,1]           (默认: 0.35)
    --pH          工作缓冲液 pH             (默认: 6.0)
    --exp         Empower 导出文件路径       (默认: data/empower_peak_results.txt)
    --component   实验数据中的目标组分名     (默认: mAb_Main_Peak)
    --threshold   收敛误差阈值 %            (默认: 2.0)
    --max-iter    最大优化迭代次数           (默认: 10)
    --no-nu-tune  禁用 nu 微调             (默认: 启用)
    --no-plot     不弹出层析图              (默认: 弹出)
    --log-level   日志级别 DEBUG/INFO/WARNING (默认: INFO)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── 路径设置 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "utils"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ProtePilot — 蛋白 IEX 层析仿真 & 参数自动校准",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 蛋白属性
    p.add_argument("--name",      default="mAb_ExampleA",          help="蛋白名称")
    p.add_argument("--pI",        type=float, default=8.5,          help="等电点")
    p.add_argument("--MW",        type=float, default=148.0,        help="分子量 (kDa)")
    p.add_argument("--hydro",     type=float, default=0.35,         help="疏水性得分 [0,1]")
    p.add_argument("--pH",        type=float, default=6.0,          help="工作 pH")
    # 实验数据
    p.add_argument("--exp",       default=None,                     help="Empower 导出文件")
    p.add_argument("--component", default="mAb_Main_Peak",          help="目标组分名")
    # 优化参数
    p.add_argument("--threshold", type=float, default=2.0,          help="收敛阈值 %%")
    p.add_argument("--max-iter",  type=int,   default=10,           help="最大迭代次数（Secant + nu tune）")
    p.add_argument("--no-nu-tune",action="store_true",              help="禁用 nu 微调阶段")
    # 输出
    p.add_argument("--no-plot",   action="store_true",              help="不弹出层析图")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"],       help="日志级别")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level   = getattr(logging, level),
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
        stream  = sys.stdout,
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("main")

    # ── 打印启动 Banner ──────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       ProtePilot — 生物制药数字孪生平台                 ║")
    print("║       IEX 离子交换层析  ·  SMA 模型  ·  自动参数校准     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ── Step 0: 导入模块 ─────────────────────────────────────────────────
    try:
        from PropertyMapper   import PropertyMapper, ProteinProperties, MapperConfig
        from MainOrchestrator import MainOrchestrator
        from cadet_engine     import ProcessParams
    except ImportError as e:
        log.error("模块导入失败：%s", e)
        log.error("请确认依赖已安装：pip install h5py numpy matplotlib")
        return 1

    # ── Step 1: 构建蛋白属性对象 ─────────────────────────────────────────
    try:
        protein = ProteinProperties(
            name           = args.name,
            pI             = args.pI,
            MW_kDa         = args.MW,
            hydrophobicity = args.hydro,
            pH_working     = args.pH,
        )
    except ValueError as e:
        log.error("蛋白属性参数错误：%s", e)
        return 1

    log.info("蛋白属性: %s", protein)

    # ── Step 2: 确定实验数据路径 ─────────────────────────────────────────
    exp_path = args.exp
    if exp_path is None:
        default_exp = ROOT / "data" / "empower_peak_results.txt"
        exp_path = str(default_exp) if default_exp.exists() else None

    if exp_path:
        log.info("实验数据: %s", exp_path)
    else:
        log.warning("未找到 Empower 实验数据，将使用 mock RT 值")

    # ── Step 3: 工艺参数（使用默认 IEX 工艺参数）───────────────────────
    proc_params = ProcessParams(
        length       = 0.25,       # 柱长 0.25 m
        col_porosity = 0.37,       # 总孔隙率
        velocity     = 5.75e-4,   # 间隙流速 m/s
        flow_rate    = 1.0e-6,    # 体积流量 m³/s
        col_disp     = 5.75e-8,   # 轴向弥散 m²/s
        salt_load    = 50.0,      # 上样盐浓度 mol/m³
        salt_elute   = 500.0,     # 洗脱终盐浓度 mol/m³
        ncol         = 100,       # 离散单元数
        nthreads     = 1,         # conda 版单线程
    )

    # ── Step 4: 初始化编排器并运行 ──────────────────────────────────────
    orch = MainOrchestrator(
        workspace     = ROOT / "data",
        engine_dir    = ROOT / "engine",
        threshold_pct = args.threshold,
        max_iter      = args.max_iter,
        nu_tune       = not args.no_nu_tune,
    )

    try:
        result = orch.run(
            protein       = protein,
            process_params= proc_params,
            exp_txt_path  = exp_path,
            exp_component = args.component,
            save_report   = True,
        )
    except Exception as e:
        log.error("运行过程中发生错误：%s", e, exc_info=True)
        return 1

    # ── Step 5: 可选绘图（最后一次仿真结果）────────────────────────────
    if not args.no_plot and result.h5_path:
        try:
            from cadet_engine import CadetEngine
            plot_engine = CadetEngine(workspace=ROOT / "data",
                                      engine_dir=ROOT / "engine")
            h5 = Path(result.h5_path)
            if h5.exists():
                t_arr, outlet = plot_engine.read_results(h5)

                import numpy as np
                from cadet_engine import SimulationResult
                sim_res = SimulationResult(
                    h5_path    = h5,
                    time       = t_arr,
                    outlet     = outlet,
                    returncode = 0,
                    wall_time  = 0.0,
                )
                plot_engine.plot_chromatogram(
                    sim_res,
                    title=f"ProtePilot — {protein.name}（校准后）",
                    show=True,
                )
        except Exception as e:
            log.warning("绘图失败（不影响主流程）：%s", e)

    # ── Step 6: 自动生成 Word 报告 ──────────────────────────────────────
    try:
        from ReportGenerator import ReportGenerator

        gen = ReportGenerator(project_root=ROOT)

        # 尝试加载执行摘要（在 reports/ 和 data/ 中查找）
        en_text, zh_text = "", ""
        for search_dir in [ROOT / "reports", ROOT / "data"]:
            en_path = search_dir / "executive_summary_EN.txt"
            if en_path.exists() and not en_text:
                en_text = en_path.read_text(encoding="utf-8")

        docx_path = gen.generate(
            protein_name        = protein.name,
            sma_params          = result.final_sma_params,
            optimization_result = result.to_dict(),
            summary_en          = en_text,
            summary_zh          = zh_text,
        )
        log.info("Word 报告已生成 → %s", docx_path)
    except Exception as e:
        log.warning("Word 报告生成失败（不影响主流程）：%s", e)

    # ── 最终状态码 ───────────────────────────────────────────────────────
    return 0 if result.converged else 2


if __name__ == "__main__":
    sys.exit(main())
