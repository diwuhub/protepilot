/**
 * bulk_report_docx.js  -  ProtePilot Bulk Analysis DOCX Report Generator
 *
 * Usage:  node src/bulk_report_docx.js <input_json> <output_docx>
 *
 * Reads a JSON file with bulk analysis results (same structure as
 * export_summary_json) and produces a professionally formatted .docx report.
 */
const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageNumber, PageBreak, LevelFormat } = require("docx");

// ── Read args ────────────────────────────────────────────────────────────────
const inputPath  = process.argv[2];
const outputPath = process.argv[3];
if (!inputPath || !outputPath) {
  console.error("Usage: node bulk_report_docx.js <input.json> <output.docx>");
  process.exit(1);
}
const data = JSON.parse(fs.readFileSync(inputPath, "utf-8"));

// ── Design tokens ────────────────────────────────────────────────────────────
const FONT = "Arial";
const C_TITLE  = "1E293B";
const C_BODY   = "334155";
const C_MUTED  = "64748B";
const C_HDR_BG = "E2E8F0";
const C_ALT    = "F8FAFC";
const C_BORDER = "CBD5E1";
const C_GREEN  = "16A34A";
const C_RED    = "DC2626";
const C_AMBER  = "D97706";

const PAGE_W = 12240;
const MARGIN = 1440;
const CW = PAGE_W - 2 * MARGIN; // 9360

const border  = { style: BorderStyle.SINGLE, size: 1, color: C_BORDER };
const borders = { top: border, bottom: border, left: border, right: border };
const pad     = { top: 60, bottom: 60, left: 100, right: 100 };

// ── Helpers ──────────────────────────────────────────────────────────────────
function txt(text, opts = {}) {
  return new TextRun({ font: FONT, size: 20, color: C_BODY, ...opts, text: String(text) });
}
function p(text, opts = {}) {
  return new Paragraph({ spacing: { after: 100 }, children: [txt(text, opts)] });
}
function hdr(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, children: [txt(text, { bold: true, size: level === HeadingLevel.HEADING_1 ? 30 : 24, color: C_TITLE })] });
}
function hCell(text, w) {
  return new TableCell({ borders, width: { size: w, type: WidthType.DXA },
    shading: { fill: C_HDR_BG, type: ShadingType.CLEAR }, margins: pad,
    children: [new Paragraph({ children: [txt(text, { bold: true, size: 18, color: C_TITLE })] })] });
}
function dCell(text, w, alt = false) {
  const sh = alt ? { fill: C_ALT, type: ShadingType.CLEAR } : undefined;
  return new TableCell({ borders, width: { size: w, type: WidthType.DXA },
    shading: sh, margins: pad,
    children: [new Paragraph({ children: [txt(text, { size: 18 })] })] });
}
function gradeColor(grade) {
  if (!grade) return C_MUTED;
  const g = String(grade).toLowerCase();
  if (g.includes("low"))  return C_GREEN;
  if (g.includes("high")) return C_RED;
  return C_AMBER;
}
function fmtNum(v, digits = 3) {
  if (v === null || v === undefined || v === "") return "-";
  return Number(v).toFixed(digits);
}

// ── Extract data ─────────────────────────────────────────────────────────────
const info    = data.batch_info || {};
const stats   = data.statistics || {};
const results = data.results    || [];

// ── Build children ───────────────────────────────────────────────────────────
const children = [];

// Title
children.push(new Paragraph({ spacing: { before: 600, after: 200 }, alignment: AlignmentType.CENTER,
  children: [txt("ProtePilot Bulk Developability Report", { bold: true, size: 36, color: C_TITLE })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
  children: [txt(`${info.molecule_class || "Unknown"} | ${info.n_total || 0} molecules | ${info.finished_at || ""}`, { size: 20, color: C_MUTED })] }));

// ── Section 1: Batch Overview ────────────────────────────────────────────────
children.push(hdr("1. Batch Overview"));

const oCols = [2400, 2400, 2280, 2280];
children.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: oCols, rows: [
  new TableRow({ children: [hCell("Metric", oCols[0]), hCell("Value", oCols[1]), hCell("Metric", oCols[2]), hCell("Value", oCols[3])] }),
  new TableRow({ children: [
    dCell("Batch Type", oCols[0], true), dCell(info.batch_type || "-", oCols[1], true),
    dCell("Molecule Class", oCols[2], true), dCell(info.molecule_class || "-", oCols[3], true) ] }),
  new TableRow({ children: [
    dCell("Total Molecules", oCols[0]), dCell(String(info.n_total || 0), oCols[1]),
    dCell("Success Rate", oCols[2]), dCell(`${info.n_success || 0}/${info.n_total || 0} (${((info.success_rate || 0) * 100).toFixed(0)}%)`, oCols[3]) ] }),
  new TableRow({ children: [
    dCell("Errors", oCols[0], true), dCell(String(info.n_error || 0), oCols[1], true),
    dCell("Wall Time", oCols[2], true), dCell(`${(info.wall_time_total_s || 0).toFixed(1)}s`, oCols[3], true) ] }),
]}));
children.push(p(""));

// ── Section 2: Per-Molecule Results ──────────────────────────────────────────
children.push(hdr("2. Per-Molecule Results"));

// Score summary line
if (stats.n_scored && stats.n_scored > 0) {
  children.push(p(`Across ${stats.n_scored} scored molecules: Mean=${fmtNum(stats.mean_score)}, Min=${fmtNum(stats.min_score)}, Max=${fmtNum(stats.max_score)} | Risk: Low=${stats.n_low_risk || 0}, Medium=${stats.n_medium_risk || 0}, High=${stats.n_high_risk || 0}`));
}
children.push(p(""));

// ── Table 2A: Biophysical Properties ────────────────────────────────────────
children.push(hdr("2a. Biophysical Properties & PTM Hotspots", HeadingLevel.HEADING_2));
const bCols = [400, 1400, 600, 900, 700, 700, 700, 700, 700, 700, 700, 660];
// Total ≈ 9360
children.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: bCols, rows: [
  new TableRow({ children: [
    hCell("#", bCols[0]), hCell("Name", bCols[1]), hCell("Len", bCols[2]),
    hCell("MW (kDa)", bCols[3]), hCell("pI", bCols[4]),
    hCell("GRAVY", bCols[5]), hCell("Deam", bCols[6]), hCell("Ox", bCols[7]),
    hCell("Cys", bCols[8]), hCell("D+E", bCols[9]), hCell("K+R+H", bCols[10]),
    hCell("Status", bCols[11]),
  ]}),
  ...results.map((r, i) => {
    const alt = i % 2 === 1;
    const vOrD = (v) => (v !== undefined && v !== "" && v !== null) ? String(v) : "-";
    return new TableRow({ children: [
      dCell(String(r.row || i + 1), bCols[0], alt),
      dCell(r.name || "-", bCols[1], alt),
      dCell(vOrD(r.seq_length), bCols[2], alt),
      dCell(fmtNum(r.mw_kda, 1), bCols[3], alt),
      dCell(fmtNum(r.pI, 2), bCols[4], alt),
      dCell(fmtNum(r.gravy, 4), bCols[5], alt),
      dCell(vOrD(r.deam_sites), bCols[6], alt),
      dCell(vOrD(r.ox_sites), bCols[7], alt),
      dCell(vOrD(r.cysteine_count), bCols[8], alt),
      dCell(vOrD(r.acidic_residues), bCols[9], alt),
      dCell(vOrD(r.basic_residues), bCols[10], alt),
      new TableCell({ borders, width: { size: bCols[11], type: WidthType.DXA },
        shading: alt ? { fill: C_ALT, type: ShadingType.CLEAR } : undefined, margins: pad,
        children: [new Paragraph({ children: [txt(r.status || "-", {
          size: 18, color: r.status === "success" ? C_GREEN : (r.status === "error" ? C_RED : C_MUTED),
        })] })] }),
    ]});
  }),
]}));
children.push(p(""));

// ── Table 2B: Developability + Risk Predictions ─────────────────────────────
children.push(hdr("2b. Developability & Risk Predictions", HeadingLevel.HEADING_2));
const rCols = [350, 1050, 650, 700, 650, 650, 650, 650, 650, 700, 700, 650, 710];
// Total ≈ 9360
children.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: rCols, rows: [
  new TableRow({ children: [
    hCell("#", rCols[0]), hCell("Name", rCols[1]),
    hCell("DevScore", rCols[2]), hCell("Base", rCols[3]), hCell("Grade", rCols[4]),
    hCell("Agg", rCols[5]), hCell("Stab", rCols[6]), hCell("Visc", rCols[7]),
    hCell("t1/2 (d)", rCols[8]), hCell("Titer", rCols[9]),
    hCell("LiabDen", rCols[10]), hCell("ADA", rCols[11]),
    hCell("OOD", rCols[12]),
  ]}),
  ...results.map((r, i) => {
    const alt = i % 2 === 1;
    const gc = gradeColor(r.dev_grade);
    return new TableRow({ children: [
      dCell(String(r.row || i + 1), rCols[0], alt),
      dCell(r.name || "-", rCols[1], alt),
      dCell(fmtNum(r.dev_score), rCols[2], alt),
      dCell(fmtNum(r.base_risk_score), rCols[3], alt),
      new TableCell({ borders, width: { size: rCols[4], type: WidthType.DXA },
        shading: alt ? { fill: C_ALT, type: ShadingType.CLEAR } : undefined, margins: pad,
        children: [new Paragraph({ children: [txt(r.dev_grade || "-", { size: 18, color: gc, bold: true })] })] }),
      dCell(fmtNum(r.agg_risk), rCols[5], alt),
      dCell(fmtNum(r.stability), rCols[6], alt),
      dCell(fmtNum(r.viscosity_risk), rCols[7], alt),
      dCell(fmtNum(r.half_life_days, 1), rCols[8], alt),
      dCell(fmtNum(r.predicted_titer_g_L, 2), rCols[9], alt),
      dCell(fmtNum(r.liability_density, 1), rCols[10], alt),
      dCell(r.ada_risk_category || "-", rCols[11], alt),
      dCell(r.ood_flag === "Y" ? "Yes" : "No", rCols[12], alt),
    ]});
  }),
]}));
children.push(p(""));

// ── Section 4: Analytical QC (cIEF & CE-SDS) ────────────────────────────────
const hasQC = results.some(r => r.cief_main_pct !== undefined && r.cief_main_pct !== "");
if (hasQC) {
  children.push(hdr("3. Analytical QC: cIEF & CE-SDS"));
  const qCols = [500, 1600, 1200, 1200, 1200, 1200, 1200, 1260];
  children.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: qCols, rows: [
    new TableRow({ children: [
      hCell("#", qCols[0]), hCell("Name", qCols[1]),
      hCell("cIEF Main%", qCols[2]), hCell("cIEF Acidic%", qCols[3]), hCell("cIEF Basic%", qCols[4]),
      hCell("CE-SDS Purity%", qCols[5]), hCell("CE-SDS HMW%", qCols[6]),
      hCell("CE-SDS LMW%", qCols[7]),
    ]}),
    ...results.filter(r => r.cief_main_pct !== undefined && r.cief_main_pct !== "").map((r, i) => {
      const alt = i % 2 === 1;
      return new TableRow({ children: [
        dCell(String(r.row || i + 1), qCols[0], alt),
        dCell(r.name || "-", qCols[1], alt),
        dCell(fmtNum(r.cief_main_pct, 1), qCols[2], alt),
        dCell(fmtNum(r.cief_acidic_pct, 1), qCols[3], alt),
        dCell(fmtNum(r.cief_basic_pct, 1), qCols[4], alt),
        dCell(fmtNum(r.ce_sds_purity_pct, 1), qCols[5], alt),
        dCell(fmtNum(r.ce_sds_hmw_pct, 1), qCols[6], alt),
        dCell(fmtNum(r.ce_sds_lmw_pct, 1), qCols[7], alt),
      ]});
    }),
  ]}));
  children.push(p(""));
}

// ── Section 5: PK & Immunogenicity ──────────────────────────────────────────
const hasPK = results.some(r => r.half_life_days !== undefined && r.half_life_days !== "");
const hasImm = results.some(r => r.ada_risk_category !== undefined && r.ada_risk_category !== "");
if (hasPK || hasImm) {
  children.push(hdr("4. Preclinical PK & Immunogenicity"));
  const pkCols = [500, 1800, 1400, 1400, 1300, 1300, 1660];
  children.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: pkCols, rows: [
    new TableRow({ children: [
      hCell("#", pkCols[0]), hCell("Name", pkCols[1]),
      hCell("Half-Life (d)", pkCols[2]), hCell("Clearance", pkCols[3]),
      hCell("ADA Risk", pkCols[4]), hCell("ADA Score", pkCols[5]),
      hCell("MHC-II Hotspots", pkCols[6]),
    ]}),
    ...results.filter(r => (r.half_life_days !== undefined && r.half_life_days !== "") || (r.ada_risk_category !== undefined && r.ada_risk_category !== "")).map((r, i) => {
      const alt = i % 2 === 1;
      return new TableRow({ children: [
        dCell(String(r.row || i + 1), pkCols[0], alt),
        dCell(r.name || "-", pkCols[1], alt),
        dCell(fmtNum(r.half_life_days, 1), pkCols[2], alt),
        dCell(fmtNum(r.clearance_ml_day_kg, 2), pkCols[3], alt),
        dCell(r.ada_risk_category || "-", pkCols[4], alt),
        dCell(fmtNum(r.ada_risk_score), pkCols[5], alt),
        dCell(r.n_mhcii_hotspots !== undefined && r.n_mhcii_hotspots !== "" ? String(r.n_mhcii_hotspots) : "-", pkCols[6], alt),
      ]});
    }),
  ]}));
  children.push(p(""));
}

// ── Section 5b: Stability Assessment ─────────────────────────────────────────
const hasStab = results.some(r => r.shelf_life_months !== undefined && r.shelf_life_months !== "");
if (hasStab) {
  children.push(hdr("5. ICH Stability Assessment"));
  const stCols = [500, 2500, 2000, 2000, 2360];
  children.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: stCols, rows: [
    new TableRow({ children: [
      hCell("#", stCols[0]), hCell("Name", stCols[1]),
      hCell("Shelf Life (mo)", stCols[2]), hCell("Stability Grade", stCols[3]),
      hCell("Liability Summary", stCols[4]),
    ]}),
    ...results.filter(r => r.shelf_life_months !== undefined && r.shelf_life_months !== "").map((r, i) => {
      const alt = i % 2 === 1;
      return new TableRow({ children: [
        dCell(String(r.row || i + 1), stCols[0], alt),
        dCell(r.name || "-", stCols[1], alt),
        dCell(fmtNum(r.shelf_life_months, 0), stCols[2], alt),
        dCell(r.stability_grade || "-", stCols[3], alt),
        dCell(r.liability_summary || "-", stCols[4], alt),
      ]});
    }),
  ]}));
  children.push(p(""));
}

// Errors section (if any)
const errors = results.filter(r => r.error);
if (errors.length > 0) {
  children.push(hdr("6. Error Details", HeadingLevel.HEADING_2));
  errors.forEach(e => {
    children.push(p(`${e.name}: ${e.error}`, { size: 18, color: C_RED }));
  });
  children.push(p(""));
}

// Footer note
children.push(new Paragraph({
  border: { top: { style: BorderStyle.SINGLE, size: 1, color: C_BORDER, space: 4 } },
  spacing: { before: 300 },
  children: [txt(`Generated by ProtePilot v32.1 on ${new Date().toISOString().slice(0, 19).replace("T", " ")}`, { size: 16, color: C_MUTED, italics: true })],
}));

// ── Assemble Document ────────────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 20, color: C_BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: C_TITLE },
        paragraph: { spacing: { before: 300, after: 150 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: C_TITLE },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: PAGE_W, height: 15840 }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } },
    },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [txt("ProtePilot Bulk Analysis Report", { size: 16, color: C_MUTED })] })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [txt("Page ", { size: 16, color: C_MUTED }), new TextRun({ font: FONT, size: 16, color: C_MUTED, children: [PageNumber.CURRENT] })] })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(outputPath, buf);
  console.log("OK:" + outputPath);
});
