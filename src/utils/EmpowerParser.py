"""
EmpowerParser.py  ·  ProtePilot
===========================================================
Waters Empower 3  'Peak Results' 导出文本解析器

功能
----
1. parse_file()          解析 Empower 导出的 .txt 报告，提取每个样品的峰数据
2. compare_with_simulation()  将 CADET 仿真峰位置与实验峰位置对比，输出误差报告
3. save_json()           将解析结果保存为结构化 JSON 文件

支持的 Empower 导出格式
-----------------------
  - 多样品报告（同一文件含多个 Sample Name 块）
  - 峰表列：# / Component Name / RT / Area / Height / % Area /
            Tailing Factor / Plate Count

使用示例
--------
    from src.utils.EmpowerParser import EmpowerParser

    parser = EmpowerParser("data/empower_peak_results.txt")
    results = parser.parse_file()
    parser.save_json("data/empower_results.json")

    # 与仿真对比
    sim_peaks = {"mAb_Main_Peak": 15.1, "Acidic_Variant_1": 8.5}
    report = parser.compare_with_simulation(sim_peaks, threshold_pct=5.0)
    print(report["summary"])
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("EmpowerParser")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构定义
# ─────────────────────────────────────────────────────────────────────────────

class EmpowerPeak:
    """代表一条峰记录"""
    __slots__ = (
        "index", "component", "rt_min", "area", "height",
        "pct_area", "tailing_factor", "plate_count",
    )

    def __init__(
        self,
        index:          int,
        component:      str,
        rt_min:         float,
        area:           float,
        height:         float,
        pct_area:       float,
        tailing_factor: Optional[float] = None,
        plate_count:    Optional[int]   = None,
    ):
        self.index          = index
        self.component      = component
        self.rt_min         = rt_min
        self.area           = area
        self.height         = height
        self.pct_area       = pct_area
        self.tailing_factor = tailing_factor
        self.plate_count    = plate_count

    def to_dict(self) -> dict:
        return {
            "index":          self.index,
            "component":      self.component,
            "rt_min":         self.rt_min,
            "area_mAU_s":     self.area,
            "height_mAU":     self.height,
            "pct_area":       self.pct_area,
            "tailing_factor": self.tailing_factor,
            "plate_count":    self.plate_count,
        }

    def __repr__(self) -> str:
        return (
            f"EmpowerPeak({self.component!r}, RT={self.rt_min:.3f} min, "
            f"%Area={self.pct_area:.2f}%)"
        )


class EmpowerSample:
    """代表一个样品的完整峰结果"""

    def __init__(
        self,
        sample_name:       str,
        sample_id:         Optional[str]   = None,
        injection_vol_uL:  Optional[float] = None,
        flow_rate_mL_min:  Optional[float] = None,
        total_area:        Optional[float] = None,
        pct_recovery:      Optional[float] = None,
    ):
        self.sample_name      = sample_name
        self.sample_id        = sample_id
        self.injection_vol_uL = injection_vol_uL
        self.flow_rate_mL_min = flow_rate_mL_min
        self.total_area       = total_area
        self.pct_recovery     = pct_recovery
        self.peaks: List[EmpowerPeak] = []

    # ── 便捷查询 ──────────────────────────────────────────────────────────

    def get_peak(self, component: str) -> Optional[EmpowerPeak]:
        """按组分名称查找峰（不区分大小写）"""
        component_lower = component.lower()
        for p in self.peaks:
            if p.component.lower() == component_lower:
                return p
        return None

    def main_peak(self) -> Optional[EmpowerPeak]:
        """返回 %Area 最大的峰"""
        if not self.peaks:
            return None
        return max(self.peaks, key=lambda p: p.pct_area)

    def to_dict(self) -> dict:
        return {
            "sample_name":       self.sample_name,
            "sample_id":         self.sample_id,
            "injection_vol_uL":  self.injection_vol_uL,
            "flow_rate_mL_min":  self.flow_rate_mL_min,
            "total_area":        self.total_area,
            "pct_recovery":      self.pct_recovery,
            "peaks":             [p.to_dict() for p in self.peaks],
        }

    def __repr__(self) -> str:
        return (
            f"EmpowerSample({self.sample_name!r}, "
            f"{len(self.peaks)} peaks)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 主解析器
# ─────────────────────────────────────────────────────────────────────────────

class EmpowerParser:
    """
    Waters Empower 3 Peak Results 导出文本解析器。

    Parameters
    ----------
    filepath : str | Path
        Empower 导出的 .txt 文件路径
    """

    # 正则表达式
    _RE_SAMPLE_NAME  = re.compile(r"^Sample Name:\s*(.+)$",        re.IGNORECASE)
    _RE_SAMPLE_ID    = re.compile(r"^Sample ID:\s*(.+)$",          re.IGNORECASE)
    _RE_INJ_VOL      = re.compile(r"Injection Volume.*?:\s*([\d.]+)", re.IGNORECASE)
    _RE_FLOW_RATE    = re.compile(r"Flow Rate.*?:\s*([\d.]+)",       re.IGNORECASE)
    _RE_TOTAL_AREA   = re.compile(r"Total Area:\s*([\d,]+\.?\d*)",   re.IGNORECASE)
    _RE_RECOVERY     = re.compile(r"% Recovery:\s*([\d.]+)",         re.IGNORECASE)
    _RE_SEPARATOR    = re.compile(r"^[-]{10,}$")
    _RE_PEAK_ROW     = re.compile(
        r"^\s*(\d+)"                # index
        r"\s+(\S+(?:\s+\S+)*?)"     # component name (可含空格)
        r"\s+([\d.]+)"              # RT
        r"\s+([\d,]+\.?\d*)"        # Area
        r"\s+([\d,]+\.?\d*)"        # Height
        r"\s+([\d.]+)"              # % Area
        r"(?:\s+([\d.]+))?"         # Tailing (optional)
        r"(?:\s+([\d,]+))?\s*$"     # Plate Count (optional)
    )

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.samples: List[EmpowerSample] = []
        self._parsed = False

    # ── 公开方法 ──────────────────────────────────────────────────────────

    def parse_file(self) -> List[EmpowerSample]:
        """
        解析 Empower 导出文件，返回 EmpowerSample 列表。

        每个 EmpowerSample 包含样品元信息和 EmpowerPeak 列表。
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"Empower 文件不存在：{self.filepath}")

        log.info("解析 Empower 报告：%s", self.filepath.name)
        text = self.filepath.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        self.samples = self._parse_lines(lines)
        self._parsed = True

        log.info(
            "解析完成：%d 个样品，共 %d 条峰记录",
            len(self.samples),
            sum(len(s.peaks) for s in self.samples),
        )
        return self.samples

    def save_json(self, output_path: str | Path) -> Path:
        """
        将解析结果保存为 JSON 文件。

        Parameters
        ----------
        output_path : str | Path
            输出 JSON 文件路径（例如 "data/empower_results.json"）
        """
        if not self._parsed:
            self.parse_file()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "source_file": str(self.filepath),
            "n_samples":   len(self.samples),
            "samples":     [s.to_dict() for s in self.samples],
        }

        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        log.info("JSON 已保存 → %s  (%.1f KB)", out.name, out.stat().st_size / 1024)
        return out

    def compare_with_simulation(
        self,
        sim_peaks:     Dict[str, float],
        exp_sample:    Optional[EmpowerSample] = None,
        threshold_pct: float = 5.0,
    ) -> dict:
        """
        将 CADET 仿真峰位置与实验峰位置对比，计算误差并判断是否合格。

        Parameters
        ----------
        sim_peaks : dict
            仿真峰字典，格式为 {组分名: 保留时间(min)}
            例如：{"mAb_Main_Peak": 15.1, "Acidic_Variant_1": 8.5}
            （来自 CADET 输出的峰检测结果）

        exp_sample : EmpowerSample, optional
            用于对比的实验样品；若为 None，使用解析结果中的第一个样品

        threshold_pct : float
            合格判断阈值（默认 5%）：误差 ≤ threshold_pct 判为 PASS

        Returns
        -------
        dict : 包含逐峰对比结果和总体 summary 的字典
        """
        if not self._parsed:
            self.parse_file()

        if exp_sample is None:
            if not self.samples:
                raise ValueError("没有可用的实验数据，请先调用 parse_file()")
            exp_sample = self.samples[0]
            log.info("使用样品 '%s' 进行对比", exp_sample.sample_name)

        comparisons: List[dict] = []
        n_pass = 0
        n_fail = 0
        n_missing = 0

        for component, sim_rt in sim_peaks.items():
            exp_peak = exp_sample.get_peak(component)

            if exp_peak is None:
                log.warning("实验数据中未找到组分：%s", component)
                comparisons.append({
                    "component":   component,
                    "sim_rt_min":  sim_rt,
                    "exp_rt_min":  None,
                    "abs_err_min": None,
                    "rel_err_pct": None,
                    "status":      "MISSING",
                })
                n_missing += 1
                continue

            abs_err = abs(sim_rt - exp_peak.rt_min)
            rel_err = (abs_err / exp_peak.rt_min) * 100.0
            status  = "PASS" if rel_err <= threshold_pct else "FAIL"

            if status == "PASS":
                n_pass += 1
            else:
                n_fail += 1

            comparisons.append({
                "component":    component,
                "sim_rt_min":   round(sim_rt, 3),
                "exp_rt_min":   round(exp_peak.rt_min, 3),
                "abs_err_min":  round(abs_err, 3),
                "rel_err_pct":  round(rel_err, 2),
                "exp_pct_area": exp_peak.pct_area,
                "status":       status,
            })

        overall = "PASS" if n_fail == 0 and n_missing == 0 else "FAIL"

        report = {
            "sample_name":   exp_sample.sample_name,
            "threshold_pct": threshold_pct,
            "n_pass":        n_pass,
            "n_fail":        n_fail,
            "n_missing":     n_missing,
            "overall":       overall,
            "comparisons":   comparisons,
            "summary":       self._build_summary(
                exp_sample.sample_name, comparisons,
                n_pass, n_fail, n_missing, threshold_pct, overall,
            ),
        }

        return report

    # ── 私有方法 ──────────────────────────────────────────────────────────

    def _parse_lines(self, lines: List[str]) -> List[EmpowerSample]:
        """逐行状态机解析 Empower 报告"""
        samples: List[EmpowerSample] = []
        current_sample: Optional[EmpowerSample] = None
        in_peak_table   = False
        separator_count = 0

        for line in lines:
            stripped = line.strip()

            # ── 样品头部信息 ──────────────────────────────────────────────
            m = self._RE_SAMPLE_NAME.match(stripped)
            if m:
                if current_sample is not None:
                    samples.append(current_sample)
                current_sample = EmpowerSample(sample_name=m.group(1).strip())
                in_peak_table  = False
                separator_count = 0
                continue

            if current_sample is None:
                continue

            m = self._RE_SAMPLE_ID.match(stripped)
            if m:
                current_sample.sample_id = m.group(1).strip()
                continue

            m = self._RE_INJ_VOL.search(stripped)
            if m and current_sample.injection_vol_uL is None:
                current_sample.injection_vol_uL = float(m.group(1))
                continue

            m = self._RE_FLOW_RATE.search(stripped)
            if m and current_sample.flow_rate_mL_min is None:
                current_sample.flow_rate_mL_min = float(m.group(1))
                continue

            # ── 总面积 / 回收率（峰表底部）────────────────────────────────
            m_area = self._RE_TOTAL_AREA.search(stripped)
            m_rec  = self._RE_RECOVERY.search(stripped)
            if m_area:
                current_sample.total_area = float(m_area.group(1).replace(",", ""))
            if m_rec:
                current_sample.pct_recovery = float(m_rec.group(1))
                in_peak_table = False
                separator_count = 0
                continue

            # ── 分隔线：控制峰表状态 ──────────────────────────────────────
            if self._RE_SEPARATOR.match(stripped):
                separator_count += 1
                if separator_count == 1:
                    in_peak_table = True   # 第一条分隔线 → 进入峰表
                elif separator_count == 2:
                    in_peak_table = False  # 第二条分隔线 → 离开峰表
                continue

            # ── 峰表头部行（跳过）─────────────────────────────────────────
            if in_peak_table and stripped.startswith("#"):
                continue

            # ── 峰数据行 ──────────────────────────────────────────────────
            if in_peak_table:
                peak = self._parse_peak_row(stripped)
                if peak is not None:
                    current_sample.peaks.append(peak)

        # 末尾样品
        if current_sample is not None:
            samples.append(current_sample)

        return samples

    def _parse_peak_row(self, line: str) -> Optional[EmpowerPeak]:
        """解析单行峰数据，解析失败返回 None"""
        m = self._RE_PEAK_ROW.match(line)
        if not m:
            return None

        try:
            return EmpowerPeak(
                index          = int(m.group(1)),
                component      = m.group(2).strip(),
                rt_min         = float(m.group(3)),
                area           = float(m.group(4).replace(",", "")),
                height         = float(m.group(5).replace(",", "")),
                pct_area       = float(m.group(6)),
                tailing_factor = float(m.group(7)) if m.group(7) else None,
                plate_count    = int(m.group(8).replace(",", "")) if m.group(8) else None,
            )
        except (ValueError, AttributeError) as e:
            log.debug("峰行解析跳过：%s  (%s)", line[:60], e)
            return None

    @staticmethod
    def _build_summary(
        sample_name: str,
        comparisons: List[dict],
        n_pass: int,
        n_fail: int,
        n_missing: int,
        threshold: float,
        overall: str,
    ) -> str:
        lines = [
            "=" * 58,
            "  仿真 vs 实验  对比报告",
            "=" * 58,
            f"  样品        : {sample_name}",
            f"  判断阈值    : ±{threshold:.1f}%",
            f"  总体结论    : {'✅ PASS' if overall == 'PASS' else '❌ FAIL'}",
            f"  PASS / FAIL / 缺失 : {n_pass} / {n_fail} / {n_missing}",
            "-" * 58,
            f"  {'组分':<22} {'仿真RT':>8} {'实验RT':>8} {'误差%':>8}  {'结论':>6}",
            "-" * 58,
        ]
        for c in comparisons:
            if c["rel_err_pct"] is not None:
                lines.append(
                    f"  {c['component']:<22} "
                    f"{c['sim_rt_min']:>8.3f} "
                    f"{c['exp_rt_min']:>8.3f} "
                    f"{c['rel_err_pct']:>7.2f}%  "
                    f"{'✅' if c['status']=='PASS' else '❌'} {c['status']}"
                )
            else:
                lines.append(
                    f"  {c['component']:<22} "
                    f"{c['sim_rt_min']:>8.3f} "
                    f"{'--':>8} "
                    f"{'--':>8}  ⚠️  MISSING"
                )
        lines.append("=" * 58)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# __main__：可直接运行的演示
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 路径相对于项目根目录
    root = Path(__file__).resolve().parent.parent.parent
    txt_path  = root / "data" / "empower_peak_results.txt"
    json_path = root / "data" / "empower_results.json"

    # ── 1. 解析 ──────────────────────────────────────────────────────────
    parser  = EmpowerParser(txt_path)
    samples = parser.parse_file()

    print(f"\n解析到 {len(samples)} 个样品：")
    for s in samples:
        print(f"  {s}")
        for p in s.peaks:
            print(f"    {p}")

    # ── 2. 保存 JSON ─────────────────────────────────────────────────────
    parser.save_json(json_path)

    # ── 3. 与仿真对比（模拟 CADET 仿真峰位置）────────────────────────────
    # 实际使用时，这些值来自 CadetEngine.read_results() 的峰检测
    sim_peaks_example = {
        "mAb_Main_Peak":    15.1,    # 仿真预测 RT (min)
        "Acidic_Variant_1": 8.8,     # 略有偏差，测试 PASS/FAIL
        "Acidic_Variant_2": 10.6,    # 偏差 > 5%，预期 FAIL
        "Basic_Variant_1":  21.7,
    }

    report = parser.compare_with_simulation(
        sim_peaks=sim_peaks_example,
        threshold_pct=5.0,
    )

    print("\n" + report["summary"])
    print(f"\n完整报告结构键：{list(report.keys())}")
