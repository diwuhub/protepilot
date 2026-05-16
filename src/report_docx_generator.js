/**
 * report_docx_generator.js — PharmaDev AI Global Report (DOCX)
 * ==============================================================
 * Reads a JSON ReportObject from stdin or file, renders to professional DOCX.
 *
 * Usage:
 *   node src/report_docx_generator.js <input.json> <output.docx>
 *
 * The DOCX follows a fixed 8-section structure:
 *   1. Executive Summary (front page)
 *   2. Molecule Overview
 *   3. Developability Assessment
 *   4. Analytical Summary
 *   5. Process / PK Summary
 *   6. Validation Plan
 *   7. Model & Confidence Metadata
 *   8. Appendix
 */

const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageBreak, PageNumber, LevelFormat,
} = require("docx");

// ── Colors ─────────────────────────────────────────────────────────
const BRAND = "1E3A5F";      // Dark navy
const ACCENT = "2E75B6";     // Medium blue
const GREEN = "16A34A";
const YELLOW = "D97706";
const RED = "DC2626";
const GRAY = "6B7280";
const LIGHT_BG = "F0F4F8";
const TABLE_HEAD = "D5E8F0";
const TABLE_ALT = "F8FAFC";

// ── Borders ────────────────────────────────────────────────────────
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };
const noBorders = {
  top: { style: BorderStyle.NONE, size: 0 },
  bottom: { style: BorderStyle.NONE, size: 0 },
  left: { style: BorderStyle.NONE, size: 0 },
  right: { style: BorderStyle.NONE, size: 0 },
};

// ── Table helpers ──────────────────────────────────────────────────
const TABLE_W = 9360; // US Letter content width at 1" margins
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: TABLE_HEAD, type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: true, font: "Arial", size: 18, color: BRAND })],
    })],
  });
}

function dataCell(text, width, opts = {}) {
  const shading = opts.shading ? { fill: opts.shading, type: ShadingType.CLEAR } : undefined;
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading,
    margins: cellMargins,
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({
        text: text || "—",
        font: "Arial", size: 18,
        bold: opts.bold || false,
        color: opts.color || "333333",
      })],
    })],
  });
}

function simpleTable(headers, rows, colWidths) {
  const tw = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: tw, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map((h, i) => headerCell(h, colWidths[i])) }),
      ...rows.map((row, ri) =>
        new TableRow({
          children: row.map((cell, ci) => {
            const isObj = typeof cell === "object" && cell !== null && !Array.isArray(cell);
            const text = isObj ? cell.text : String(cell ?? "—");
            const opts = isObj ? cell : {};
            if (ri % 2 === 1 && !opts.shading) opts.shading = TABLE_ALT;
            return dataCell(text, colWidths[ci], opts);
          }),
        })
      ),
    ],
  });
}

// ── Text helpers ───────────────────────────────────────────────────
function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, children: [new TextRun(text)] });
}

function bodyText(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({
      text,
      font: "Arial", size: 20,
      bold: opts.bold || false,
      italics: opts.italic || false,
      color: opts.color || "333333",
    })],
  });
}

function labelValue(label, value) {
  return new Paragraph({
    spacing: { after: 80 },
    children: [
      new TextRun({ text: label + ": ", font: "Arial", size: 20, bold: true, color: BRAND }),
      new TextRun({ text: String(value ?? "—"), font: "Arial", size: 20 }),
    ],
  });
}

function spacer() {
  return new Paragraph({ spacing: { after: 80 }, children: [] });
}

// ── Grade color ────────────────────────────────────────────────────
function gradeColor(grade) {
  const g = (grade || "").toLowerCase();
  if (g.includes("low") || g.includes("proceed") && !g.includes("caution")) return GREEN;
  if (g.includes("medium") || g.includes("moderate") || g.includes("caution")) return YELLOW;
  if (g.includes("high") || g.includes("optimize")) return RED;
  return GRAY;
}

function statusShading(status) {
  const s = (status || "").toLowerCase();
  if (s.includes("within target")) return "E8F5E9";
  if (s.includes("within range")) return "FFF8E1";
  if (s.includes("out of range")) return "FFEBEE";
  return "F5F5F5";
}

// ====================================================================
// SECTION BUILDERS
// ====================================================================

function buildCoverPage(r) {
  const es = r.executive_summary;
  const recColor = gradeColor(es.recommendation);
  const children = [
    spacer(), spacer(),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 40 },
      children: [new TextRun({ text: "PharmaDev AI", font: "Arial", size: 36, bold: true, color: BRAND })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: "Global Analysis Report", font: "Arial", size: 28, color: ACCENT })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 80 },
      children: [new TextRun({ text: es.molecule_name || "Unknown", font: "Arial", size: 44, bold: true, color: BRAND })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: es.molecule_class_display || "", font: "Arial", size: 24, color: GRAY })],
    }),
    // Score box
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 40 },
      children: [new TextRun({ text: `Developability Score: ${(es.overall_score || 0).toFixed(2)}`, font: "Arial", size: 28, bold: true, color: gradeColor(es.overall_grade) })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 40 },
      children: [new TextRun({ text: es.overall_grade || "", font: "Arial", size: 22, color: gradeColor(es.overall_grade) })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 200 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: ACCENT } },
      children: [new TextRun({ text: `Recommendation: ${es.recommendation || ""}`, font: "Arial", size: 24, bold: true, color: recColor })],
    }),
    bodyText(es.recommendation_detail || ""),
    spacer(),
    // Top Risks
    heading("Key Risks", HeadingLevel.HEADING_2),
  ];

  (es.top_risks || []).forEach(risk => {
    children.push(bodyText(`  \u2022  ${risk}`, { color: RED }));
  });

  if ((es.top_strengths || []).length > 0) {
    children.push(heading("Key Strengths", HeadingLevel.HEADING_2));
    es.top_strengths.forEach(s => {
      children.push(bodyText(`  \u2022  ${s}`, { color: GREEN }));
    });
  }

  if ((es.key_caveats || []).length > 0) {
    children.push(heading("Caveats", HeadingLevel.HEADING_2));
    es.key_caveats.forEach(c => {
      children.push(bodyText(`  \u2022  ${c}`, { color: GRAY, italic: true }));
    });
  }

  children.push(spacer());
  children.push(new Paragraph({
    alignment: AlignmentType.RIGHT, spacing: { before: 200 },
    children: [new TextRun({ text: `Generated: ${r.generated_at || ""}  |  ${r.platform_version || ""}`, font: "Arial", size: 16, color: GRAY })],
  }));

  return children;
}


function buildMoleculeOverview(r) {
  const mo = r.molecule_overview;
  const children = [
    new Paragraph({ children: [new PageBreak()] }),
    heading("2. Molecule Overview"),
    spacer(),
  ];

  // Key metrics table
  const metricsRows = [
    ["Molecule Name", mo.name || "—"],
    ["Molecule Class", mo.molecule_class ? mo.molecule_class.replace(/_/g, " ") : "—"],
    ["Stoichiometry", mo.stoichiometry || "—"],
    ["Molecular Weight", mo.molecular_weight_kda ? `${mo.molecular_weight_kda.toFixed(1)} kDa` : "—"],
    ["Isoelectric Point (pI)", mo.isoelectric_point ? mo.isoelectric_point.toFixed(2) : "—"],
    ["GRAVY Score", mo.gravy_score != null ? mo.gravy_score.toFixed(3) : "—"],
    ["Hydrophobicity (norm.)", mo.hydrophobicity_normalized != null ? mo.hydrophobicity_normalized.toFixed(3) : "—"],
    ["Sequence Length", `${mo.sequence_length || 0} aa`],
    ["Cysteine Count", `${mo.cysteine_count || 0}`],
    ["Fc Region", mo.has_fc_region ? "Yes" : "No"],
    ["N-Glycosylation Expected", mo.expects_glycosylation ? "Yes" : "No"],
  ];

  children.push(simpleTable(["Property", "Value"], metricsRows, [4680, 4680]));

  // Chain composition
  if ((mo.chain_composition || []).length > 0) {
    children.push(spacer());
    children.push(heading("Chain Composition", HeadingLevel.HEADING_2));
    const chainRows = mo.chain_composition.map(c => [c.name || "", c.type || "", `${c.length || 0} aa`]);
    children.push(simpleTable(["Chain", "Type", "Length"], chainRows, [3120, 3120, 3120]));
  }

  return children;
}


function buildDevelopability(r) {
  const ds = r.developability;
  const children = [
    new Paragraph({ children: [new PageBreak()] }),
    heading("3. Developability Assessment"),
    spacer(),
    labelValue("Composite Score", (ds.composite_score || 0).toFixed(3)),
    labelValue("Grade", ds.composite_grade || "—"),
    labelValue("Recommendation", ds.recommendation || "—"),
    spacer(),
  ];

  // Risk dimensions table
  if ((ds.risk_dimensions || []).length > 0) {
    children.push(heading("Risk Dimensions", HeadingLevel.HEADING_2));

    const dimRows = ds.risk_dimensions.map(d => [
      d.dimension,
      { text: (d.score || 0).toFixed(3), center: true },
      { text: d.grade || "—", center: true, bold: true, color: gradeColor(d.grade) },
      { text: `${((d.weight || 0) * 100).toFixed(0)}%`, center: true },
      d.explanation || "",
    ]);
    children.push(simpleTable(
      ["Dimension", "Score", "Grade", "Weight", "Explanation"],
      dimRows,
      [1500, 900, 900, 900, 5160],
    ));

    // Primary drivers detail
    children.push(spacer());
    ds.risk_dimensions.forEach(d => {
      if ((d.primary_drivers || []).length > 0) {
        children.push(new Paragraph({
          spacing: { after: 60 },
          children: [
            new TextRun({ text: `${d.dimension}: `, font: "Arial", size: 18, bold: true, color: BRAND }),
            new TextRun({ text: d.primary_drivers.join("; "), font: "Arial", size: 18, color: GRAY }),
          ],
        }));
      }
    });
  }

  // Liability summary
  if (ds.liability_summary && Object.keys(ds.liability_summary).length > 0) {
    children.push(spacer());
    children.push(heading("Liability Summary", HeadingLevel.HEADING_2));
    const liabRows = Object.entries(ds.liability_summary)
      .filter(([, v]) => v > 0)
      .map(([k, v]) => [k.replace(/_/g, " "), { text: String(v), center: true }]);
    if (liabRows.length > 0) {
      children.push(simpleTable(["Liability Type", "Count"], liabRows, [6000, 3360]));
    }
  }

  // QTPP table
  if ((ds.qtpp_rows || []).length > 0) {
    children.push(spacer());
    children.push(heading("Quality Target Product Profile (QTPP)", HeadingLevel.HEADING_2));
    children.push(bodyText("ICH Q8(R2) framework — Critical Quality Attributes with platform predictions.", { italic: true, color: GRAY }));

    const qtppRows = ds.qtpp_rows.map(q => [
      q.attribute,
      { text: q.target, center: true },
      { text: q.acceptable_range, center: true },
      { text: q.current_prediction, center: true, bold: true },
      { text: q.status, center: true, shading: statusShading(q.status) },
    ]);
    children.push(simpleTable(
      ["CQA", "Target", "Acceptable Range", "Current", "Status"],
      qtppRows,
      [2400, 1600, 1600, 1600, 2160],
    ));
  }

  return children;
}


function buildAnalytical(r) {
  const an = r.analytical;
  const children = [
    new Paragraph({ children: [new PageBreak()] }),
    heading("4. Analytical / Characterization Summary"),
    spacer(),
  ];

  const rows = [];
  if (an.sec_monomer_pct != null) rows.push(["SEC Monomer Purity", `${an.sec_monomer_pct.toFixed(1)}%`]);
  if (an.sec_hmw_pct != null) rows.push(["SEC HMW (Aggregates)", `${an.sec_hmw_pct.toFixed(1)}%`]);
  if (an.cesds_intact_pct != null) rows.push(["CE-SDS Intact IgG", `${an.cesds_intact_pct.toFixed(1)}%`]);
  if (an.cief_main_pct != null) rows.push(["cIEF Main Peak", `${an.cief_main_pct.toFixed(1)}%`]);
  if (an.cief_acidic_pct != null) rows.push(["cIEF Acidic Variants", `${an.cief_acidic_pct.toFixed(1)}%`]);
  if (an.cief_basic_pct != null) rows.push(["cIEF Basic Variants", `${an.cief_basic_pct.toFixed(1)}%`]);
  if (an.ms_intact_mass_da != null) rows.push(["Intact Mass", `${an.ms_intact_mass_da.toFixed(1)} Da`]);

  if (rows.length > 0) {
    children.push(simpleTable(["Assay", "Result"], rows, [4680, 4680]));
  } else {
    children.push(bodyText("Analytical QC data not yet available for this analysis session.", { italic: true, color: GRAY }));
  }

  if (an.purity_note) { children.push(spacer()); children.push(bodyText(an.purity_note)); }
  if (an.charge_variant_note) { children.push(bodyText(an.charge_variant_note)); }
  if (an.ms_note) { children.push(bodyText(an.ms_note)); }

  return children;
}


function buildProcessPK(r) {
  const pk = r.process_pk;
  const children = [
    new Paragraph({ children: [new PageBreak()] }),
    heading("5. Process / PK / Downstream Summary"),
    spacer(),
  ];

  // PK
  if (pk.half_life_days != null) {
    children.push(heading("Pharmacokinetics", HeadingLevel.HEADING_2));
    children.push(labelValue("Predicted Half-Life", `${pk.half_life_days.toFixed(1)} days`));
    children.push(labelValue("PK Risk Level", pk.pk_risk_level || "—"));
    if (pk.pk_note) children.push(bodyText(pk.pk_note));
    children.push(spacer());
  }

  // Upstream
  if (pk.final_titer_g_l != null) {
    children.push(heading("Upstream Bioreactor", HeadingLevel.HEADING_2));
    children.push(labelValue("Predicted Final Titer", `${pk.final_titer_g_l.toFixed(2)} g/L`));
    if (pk.upstream_note) children.push(bodyText(pk.upstream_note));
    children.push(spacer());
  }

  // ADA
  if (pk.ada_risk_level) {
    children.push(heading("Immunogenicity (ADA)", HeadingLevel.HEADING_2));
    children.push(labelValue("ADA Risk Level", pk.ada_risk_level));
    if (pk.ada_note) children.push(bodyText(pk.ada_note));
    children.push(spacer());
  }

  // Chromatography
  if (pk.cex_summary) {
    children.push(heading("Chromatography", HeadingLevel.HEADING_2));
    children.push(bodyText(pk.cex_summary));
  }

  if (!pk.half_life_days && !pk.final_titer_g_l && !pk.ada_risk_level && !pk.cex_summary) {
    children.push(bodyText("Process and PK data not yet available for this analysis session.", { italic: true, color: GRAY }));
  }

  return children;
}


function buildValidationPlan(r) {
  const vp = r.validation_plan;
  const children = [
    new Paragraph({ children: [new PageBreak()] }),
    heading("6. Validation Plan"),
    spacer(),
    labelValue("Total Recommended Assays", String(vp.total_assays || 0)),
  ];

  if (vp.molecule_class_impact) {
    children.push(bodyText(vp.molecule_class_impact, { italic: true, color: ACCENT }));
    children.push(spacer());
  }

  // Required assays
  if ((vp.required_assays || []).length > 0) {
    children.push(heading("Required Assays (ICH Q6B)", HeadingLevel.HEADING_2));
    const rows = vp.required_assays.map(a => [a.name, { text: a.priority, center: true }, a.reason || ""]);
    children.push(simpleTable(["Assay", "Priority", "Reason"], rows, [3500, 1200, 4660]));
    children.push(spacer());
  }

  // Format-specific
  if ((vp.format_specific_assays || []).length > 0) {
    children.push(heading("Format-Specific Assays", HeadingLevel.HEADING_2));
    const rows = vp.format_specific_assays.map(a => [
      a.name, { text: a.priority, center: true },
      a.explanation || a.reason || "",
    ]);
    children.push(simpleTable(["Assay", "Priority", "Why Required"], rows, [3200, 1200, 4960]));
    children.push(spacer());
  }

  // Risk-triggered
  if ((vp.risk_triggered_assays || []).length > 0) {
    children.push(heading("Risk-Triggered Assays", HeadingLevel.HEADING_2));
    const rows = vp.risk_triggered_assays.map(a => [a.name, { text: a.priority, center: true }, a.reason || ""]);
    children.push(simpleTable(["Assay", "Priority", "Trigger"], rows, [3500, 1200, 4660]));
    children.push(spacer());
  }

  // Recommendations
  if ((vp.key_recommendations || []).length > 0) {
    children.push(heading("Recommendations", HeadingLevel.HEADING_2));
    vp.key_recommendations.forEach(rec => {
      children.push(bodyText(`  \u2022  ${rec}`));
    });
  }

  return children;
}


function buildModelMetadata(r) {
  const mm = r.model_metadata;
  const children = [
    new Paragraph({ children: [new PageBreak()] }),
    heading("7. Confidence & Limitations"),
    spacer(),
  ];

  // Route summary table
  const routeRows = [
    ["Analysis Route", mm.analysis_route || "—"],
    ["Model Source", mm.model_source || "—"],
    ["Embedding", mm.embedding_mode || "—"],
    ["Prediction Type", mm.heuristic_ml_hybrid || "—"],
    ["Confidence Level", mm.confidence_level || "—"],
    ["OOD Flag", mm.is_ood ? `Yes — ${mm.ood_reason}` : "No"],
  ];
  children.push(simpleTable(["Parameter", "Value"], routeRows, [3120, 6240]));
  children.push(spacer());

  if (mm.confidence_rationale) {
    children.push(heading("Confidence Rationale", HeadingLevel.HEADING_2));
    children.push(bodyText(mm.confidence_rationale));
    children.push(spacer());
  }

  if (mm.molecule_class_benchmark_note) {
    children.push(heading("Benchmark Coverage", HeadingLevel.HEADING_2));
    children.push(bodyText(mm.molecule_class_benchmark_note));
    children.push(spacer());
  }

  // High confidence conclusions
  if ((mm.high_confidence_conclusions || []).length > 0) {
    children.push(heading("High-Confidence Conclusions", HeadingLevel.HEADING_2));
    mm.high_confidence_conclusions.forEach(c => {
      children.push(bodyText(`  \u2713  ${c}`, { color: GREEN }));
    });
    children.push(spacer());
  }

  // Low confidence conclusions
  if ((mm.low_confidence_conclusions || []).length > 0) {
    children.push(heading("Lower-Confidence Estimates", HeadingLevel.HEADING_2));
    mm.low_confidence_conclusions.forEach(c => {
      children.push(bodyText(`  \u26A0  ${c}`, { color: YELLOW }));
    });
    children.push(spacer());
  }

  // Caveats
  if ((mm.caveats || []).length > 0) {
    children.push(heading("Caveats", HeadingLevel.HEADING_2));
    mm.caveats.forEach(c => {
      children.push(bodyText(`  \u2022  ${c}`, { italic: true, color: RED }));
    });
  }

  return children;
}


function buildAppendix(r) {
  const ap = r.appendix;
  const children = [
    new Paragraph({ children: [new PageBreak()] }),
    heading("8. Appendix — Selected Raw Metrics"),
    spacer(),
  ];

  // Biophysical features
  const feats = ap.biophysical_features || {};
  if (Object.keys(feats).length > 0) {
    children.push(heading("Biophysical Features", HeadingLevel.HEADING_2));
    const rows = Object.entries(feats).map(([k, v]) => [
      k.replace(/_/g, " "),
      { text: typeof v === "number" ? v.toFixed(4) : String(v), center: true },
    ]);
    children.push(simpleTable(["Feature", "Value"], rows, [5000, 4360]));
    children.push(spacer());
  }

  // CDR regions
  if ((ap.cdr_regions || []).length > 0) {
    children.push(heading("CDR Regions", HeadingLevel.HEADING_2));
    const rows = ap.cdr_regions.map(c => [c.chain, c.cdr, c.sequence, { text: String(c.length), center: true }]);
    children.push(simpleTable(["Chain", "CDR", "Sequence", "Length"], rows, [1500, 1500, 4860, 1500]));
  }

  return children;
}


// ====================================================================
// MAIN: Build Document
// ====================================================================

async function generateReport(inputPath, outputPath) {
  const raw = fs.readFileSync(inputPath, "utf-8");
  const r = JSON.parse(raw);

  const allChildren = [
    ...buildCoverPage(r),
    ...buildMoleculeOverview(r),
    ...buildDevelopability(r),
    ...buildAnalytical(r),
    ...buildProcessPK(r),
    ...buildValidationPlan(r),
    ...buildModelMetadata(r),
    ...buildAppendix(r),
  ];

  const doc = new Document({
    styles: {
      default: { document: { run: { font: "Arial", size: 20 } } },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
          quickFormat: true,
          run: { size: 32, bold: true, font: "Arial", color: BRAND },
          paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 0 },
        },
        {
          id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
          quickFormat: true,
          run: { size: 24, bold: true, font: "Arial", color: ACCENT },
          paragraph: { spacing: { before: 180, after: 120 }, outlineLevel: 1 },
        },
      ],
    },
    numbering: {
      config: [{
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      }],
    },
    sections: [{
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: ACCENT, space: 4 } },
            children: [
              new TextRun({ text: `PharmaDev AI  |  ${r.executive_summary?.molecule_name || "Report"}`, font: "Arial", size: 16, color: GRAY }),
            ],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            border: { top: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC", space: 4 } },
            children: [
              new TextRun({ text: "Page ", font: "Arial", size: 16, color: GRAY }),
              new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: GRAY }),
              new TextRun({ text: `    |    ${r.generated_at || ""}`, font: "Arial", size: 16, color: GRAY }),
            ],
          })],
        }),
      },
      children: allChildren,
    }],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(outputPath, buffer);
  console.log(`DOCX written: ${outputPath} (${buffer.length} bytes)`);
}

// ── CLI ─────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
if (args.length < 2) {
  console.error("Usage: node report_docx_generator.js <input.json> <output.docx>");
  process.exit(1);
}

generateReport(args[0], args[1]).catch(err => {
  console.error("Error:", err);
  process.exit(1);
});
