"""
ProtePilot — Unified Color & Design Token System
====================================================
All status indicators, risk flags, semantic colors, and design tokens
are defined here. No emojis — color-coded HTML spans used throughout.

Usage:
    from src.ui_colors import STATUS_DOT, status_badge, COLORS, T

    st.markdown(STATUS_DOT["pass"] + " Within Target", unsafe_allow_html=True)
    st.markdown(status_badge("Low Risk", "pass"), unsafe_allow_html=True)
"""

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS  — Single source of truth for the entire UI
# ══════════════════════════════════════════════════════════════════════════════

class T:
    """Design tokens.  Import as `from src.ui_colors import T`."""

    # ── Text ─────────────────────────────────────────────────────────────
    TEXT_DARK       = "#0F172A"   # Headings, hero numbers
    TEXT_PRIMARY    = "#334155"   # Body text
    TEXT_SECONDARY  = "#64748B"   # Subtitles, captions
    TEXT_MUTED      = "#94A3B8"   # Hints, units, disabled

    # ── Semantic Status (brighter set) ───────────────────────────────────
    PASS            = "#10B981"   # Green  — success / within target
    CAUTION         = "#F59E0B"   # Amber  — warning / within range
    FAIL            = "#EF4444"   # Red    — error / out of range
    INFO            = "#3B82F6"   # Blue   — informational
    NEUTRAL         = "#6B7280"   # Gray   — not assessed / neutral

    # ── Status backgrounds (light tint) ──────────────────────────────────
    PASS_BG         = "#F0FFF4"
    CAUTION_BG      = "#FFFBEB"
    FAIL_BG         = "#FEF2F2"
    INFO_BG         = "#F0F4FF"
    NEUTRAL_BG      = "#F9FAFB"

    # ── Surfaces ─────────────────────────────────────────────────────────
    BG              = "#FFFFFF"   # Page / card background
    PANEL_BG        = "#F8FAFC"   # Sidebar, subtle panels
    INPUT_BG        = "#F1F5F9"   # Input field background
    HOVER_BG        = "#F8FAFC"   # Button / row hover
    ACTIVE_BG       = "#E2E8F0"   # Button pressed

    # ── Borders ──────────────────────────────────────────────────────────
    BORDER          = "#E2E8F0"   # Standard card / divider border
    BORDER_BTN      = "#CBD5E1"   # Button borders (slightly darker)
    BORDER_FOCUS    = "#0F172A"   # Input focus ring
    BORDER_HOVER    = "#94A3B8"   # Input / button hover

    # ── Accent ───────────────────────────────────────────────────────────
    PURPLE          = "#7C3AED"   # AI / training highlights
    PURPLE_BG       = "#F5F3FF"

    # ── Border Radius (3-tier + pill) ────────────────────────────────────
    RADIUS_SM       = "6px"      # Buttons, badges, small tags
    RADIUS_MD       = "8px"      # Standard cards, panels
    RADIUS_LG       = "12px"     # Hero cards, metric displays
    RADIUS_PILL     = "20px"     # Status pills

    # ── Border-Left Widths ───────────────────────────────────────────────
    BL_STANDARD     = "4px"      # Normal info / status cards
    BL_CRITICAL     = "5px"      # Composite risk banners only

    # ── Font Sizes (rem) ─────────────────────────────────────────────────
    FS_HERO         = "2.8rem"   # Big display numbers (pI, MW)
    FS_H1           = "1.6rem"   # Page titles (h2)
    FS_H2           = "1.15rem"  # Section headers (h3)
    FS_H3           = "1.0rem"   # Card titles, sub-sections
    FS_BODY         = "0.9rem"   # Body text
    FS_LABEL        = "0.85rem"  # Labels, badge text
    FS_SMALL        = "0.8rem"   # Captions, helper text
    FS_XS           = "0.75rem"  # Fine print, axis labels

    # ── Font Weights ─────────────────────────────────────────────────────
    FW_REGULAR      = "400"
    FW_SEMIBOLD     = "600"
    FW_BOLD         = "700"

    # ── Shadows ──────────────────────────────────────────────────────────
    SHADOW_SM       = "0 1px 3px rgba(0,0,0,0.04)"
    SHADOW_MD       = "0 4px 10px rgba(0,0,0,0.06)"
    SHADOW_LG       = "0 4px 16px rgba(0,0,0,0.12)"
    SHADOW_CARD     = "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02)"


# ══════════════════════════════════════════════════════════════════════════════
#  LEGACY-COMPATIBLE EXPORTS — kept so nothing breaks
# ══════════════════════════════════════════════════════════════════════════════

# ── Core Palette (dict form) ─────────────────────────────────────────────────
COLORS = {
    "pass":    {"primary": T.PASS,    "bg": T.PASS_BG,    "border": T.PASS},
    "caution": {"primary": T.CAUTION, "bg": T.CAUTION_BG, "border": T.CAUTION},
    "fail":    {"primary": T.FAIL,    "bg": T.FAIL_BG,    "border": T.FAIL},
    "info":    {"primary": T.INFO,    "bg": T.INFO_BG,    "border": T.INFO},
    "neutral": {"primary": T.NEUTRAL, "bg": T.NEUTRAL_BG, "border": T.NEUTRAL},
}

# ── Enterprise constants (now aliases into T) ────────────────────────────────
SLATE       = T.TEXT_SECONDARY
DARK        = T.TEXT_DARK
ACCENT      = T.TEXT_PRIMARY
SUCCESS     = T.PASS
WARN        = T.CAUTION
ERROR       = T.FAIL
BG          = T.BG
CARD_BG     = T.BG
SIDEBAR_BG  = T.PANEL_BG
SIDEBAR_TEXT = T.TEXT_PRIMARY
BORDER      = T.BORDER

# ── Status Dots ──────────────────────────────────────────────────────────────
_DOT_TEMPLATE = (
    '<span style="display:inline-block;width:10px;height:10px;'
    'border-radius:50%;background:{color};margin-right:6px;vertical-align:middle;">'
    '</span>'
)

STATUS_DOT = {
    "pass":    _DOT_TEMPLATE.format(color=T.PASS),
    "caution": _DOT_TEMPLATE.format(color=T.CAUTION),
    "fail":    _DOT_TEMPLATE.format(color=T.FAIL),
    "info":    _DOT_TEMPLATE.format(color=T.INFO),
    "neutral": _DOT_TEMPLATE.format(color=T.NEUTRAL),
}


# ── Status Badge ─────────────────────────────────────────────────────────────
def status_badge(text: str, level: str = "neutral") -> str:
    """Return an HTML badge with colored background."""
    c = COLORS.get(level, COLORS["neutral"])
    return (
        f'<span style="background:{c["bg"]};color:{c["primary"]};'
        f'border:1px solid {c["border"]};border-radius:{T.RADIUS_SM};'
        f'padding:2px 8px;font-size:{T.FS_LABEL};font-weight:{T.FW_SEMIBOLD};">'
        f'{text}</span>'
    )


# ── Colored Text ─────────────────────────────────────────────────────────────
def colored_text(text: str, level: str = "neutral", bold: bool = True) -> str:
    """Return HTML span with colored text."""
    c = COLORS.get(level, COLORS["neutral"])
    weight = f"font-weight:{T.FW_SEMIBOLD};" if bold else ""
    return f'<span style="color:{c["primary"]};{weight}">{text}</span>'


def colored_cell(text: str, level: str = "neutral") -> str:
    """Return HTML span with colored background (table cell highlight)."""
    c = COLORS.get(level, COLORS["neutral"])
    return (
        f'<span style="background:{c["bg"]};color:{c["primary"]};'
        f'border-radius:{T.RADIUS_SM};padding:1px 6px;'
        f'font-weight:{T.FW_SEMIBOLD};font-size:{T.FS_LABEL};">'
        f'{text}</span>'
    )


# Map common status strings to color levels
STATUS_LEVEL_MAP = {
    "Within Target": "pass",
    "Within Range": "caution",
    "Out of Range": "fail",
    "Not Assessed": "neutral",
    "Pass": "pass",
    "Fail": "fail",
    "Flag": "caution",
    "Low Risk": "pass",
    "Medium Risk": "caution",
    "High Risk": "fail",
    "Proceed": "pass",
    "Caution": "caution",
    "Optimize": "fail",
}


def auto_color(text: str, bold: bool = True) -> str:
    """Auto-detect status level from text and return colored HTML."""
    level = STATUS_LEVEL_MAP.get(text, "neutral")
    return colored_text(text, level, bold)


# ── QTPP / QC Status Mapping ─────────────────────────────────────────────────
QTPP_STATUS = {
    "Within Target": {"dot": STATUS_DOT["pass"],    "label": "Within Target",  "level": "pass"},
    "Within Range":  {"dot": STATUS_DOT["caution"], "label": "Within Range",   "level": "caution"},
    "Out of Range":  {"dot": STATUS_DOT["fail"],    "label": "Out of Range",   "level": "fail"},
    "Not Assessed":  {"dot": STATUS_DOT["neutral"], "label": "Not Assessed",   "level": "neutral"},
}

# ── Severity Dots ────────────────────────────────────────────────────────────
SEVERITY_DOT = {
    "critical": STATUS_DOT["fail"],
    "warning":  STATUS_DOT["caution"],
    "ok":       STATUS_DOT["pass"],
    "info":     STATUS_DOT["info"],
}


# ── Chart Color Palette ──────────────────────────────────────────────────────
CHART_COLORS = {
    "primary":   T.INFO,      # Blue
    "secondary": T.PASS,      # Green
    "accent":    T.CAUTION,   # Amber
    "danger":    T.FAIL,      # Red
    "muted":     T.NEUTRAL,   # Gray
    "purple":    "#8B5CF6",   # Purple (chart series)
}

CHART_PALETTE = [
    CHART_COLORS["primary"],
    CHART_COLORS["secondary"],
    CHART_COLORS["accent"],
    CHART_COLORS["danger"],
    CHART_COLORS["purple"],
    CHART_COLORS["muted"],
]


# ── Plotly Defaults ──────────────────────────────────────────────────────────
PLOTLY_COLORS = [
    CHART_COLORS["primary"],
    CHART_COLORS["secondary"],
    CHART_COLORS["accent"],
    CHART_COLORS["danger"],
    CHART_COLORS["purple"],
    CHART_COLORS["muted"],
    "#0891B2",                  # Cyan (extended)
    "#BE185D",                  # Magenta (extended)
]

PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=T.TEXT_PRIMARY),
    title_font=dict(size=15, color=T.TEXT_DARK, family="Inter, system-ui, sans-serif"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=48, r=24, t=56, b=48),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=T.BORDER,
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        showgrid=True, gridcolor=T.INPUT_BG, gridwidth=1,
        linecolor=T.BORDER, linewidth=1,
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=True, gridcolor=T.INPUT_BG, gridwidth=1,
        linecolor=T.BORDER, linewidth=1,
        zeroline=False,
    ),
)


# ── Selftest ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== ui_colors selftest ===")
    for level in ("pass", "caution", "fail", "info", "neutral"):
        assert level in STATUS_DOT, f"Missing STATUS_DOT[{level}]"
        assert level in COLORS, f"Missing COLORS[{level}]"
    badge = status_badge("Low Risk", "pass")
    assert "Low Risk" in badge
    assert COLORS["pass"]["primary"] in badge
    for status in QTPP_STATUS:
        assert "dot" in QTPP_STATUS[status]
    ct = colored_text("Pass", "pass")
    assert COLORS["pass"]["primary"] in ct
    cc = colored_cell("Within Target", "pass")
    assert COLORS["pass"]["bg"] in cc
    ac = auto_color("High Risk")
    assert COLORS["fail"]["primary"] in ac
    # Token consistency
    assert T.PASS == COLORS["pass"]["primary"]
    assert SUCCESS == T.PASS
    assert BORDER == T.BORDER
    print("All checks passed.")
