import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


# ============================================================
# 0. Global plotting style
# ============================================================
plt.rcParams.update({
    "figure.dpi": 135,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.6,
})


# ============================================================
# 1. Global CSS: preserve original layout, improve typography/color/spacing
# ============================================================
custom_css = """
:root {
    --ink: #172033;
    --muted: #64748b;
    --line: #dbe3ef;
    --panel: #ffffff;
    --panel-soft: #f8fafc;
    --blue: #2563eb;
    --blue-soft: #eff6ff;
    --green: #16a34a;
    --green-soft: #ecfdf5;
    --amber: #d97706;
    --amber-soft: #fff7ed;
    --red: #dc2626;
    --red-soft: #fef2f2;
    --purple: #7c3aed;
    --purple-soft: #f5f3ff;
}

.gradio-container {
    max-width: 1760px !important;
    margin: 0 auto !important;
    padding: 14px 20px 24px 20px !important;
    color: var(--ink) !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif !important;
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
}

.center-title {
    text-align: center !important;
    margin: 4px auto 18px auto !important;
}

.center-title h1 {
    font-size: 29px !important;
    font-weight: 760 !important;
    letter-spacing: -0.35px !important;
    margin-bottom: 5px !important;
    color: #0f172a !important;
}

.center-title p {
    font-size: 14.5px !important;
    color: var(--muted) !important;
    margin-top: 2px !important;
}

/* Keep the original section-based interface, but make it lighter and cleaner. */
.panel-card {
    background: rgba(255,255,255,0.96) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    padding: 15px 16px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05) !important;
    margin-bottom: 14px !important;
    text-align: center !important;
}

.section-heading {
    text-align: center !important;
    font-size: 16px !important;
    font-weight: 740 !important;
    margin: 0 auto 13px auto !important;
    color: #0f172a !important;
    padding: 10px 8px !important;
    border-radius: 8px !important;
    background: linear-gradient(90deg, #f1f5f9, #f8fafc) !important;
    border: 1px solid #e2e8f0 !important;
}

.field-label {
    text-align: center !important;
    font-size: 13.5px !important;
    font-weight: 650 !important;
    color: #334155 !important;
    margin-top: 10px !important;
    margin-bottom: 4px !important;
    line-height: 1.45 !important;
}

.field-note {
    text-align: center !important;
    font-size: 11.5px !important;
    color: #64748b !important;
    margin-top: -1px !important;
    margin-bottom: 6px !important;
    line-height: 1.55 !important;
}

.left-panel input[type='number'],
.left-panel input[type='text'] {
    text-align: center !important;
    font-size: 15px !important;
    height: 42px !important;
    border-radius: 8px !important;
    border: 1px solid #dbe3ef !important;
    background: #fbfdff !important;
    color: #0f172a !important;
}

.left-panel input[type='number']:focus,
.left-panel input[type='text']:focus {
    border-color: #93c5fd !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

.left-panel .wrap,
.left-panel label,
.left-panel label span {
    text-align: center !important;
    justify-content: center !important;
}

.big-btn {
    width: 100% !important;
    min-height: 58px !important;
    font-size: 18px !important;
    font-weight: 720 !important;
    margin-top: 18px !important;
    border-radius: 10px !important;
    letter-spacing: 0.15px !important;
    box-shadow: 0 6px 14px rgba(37,99,235,0.18) !important;
}

.output-label {
    text-align: center !important;
}

.schematic-box {
    text-align: center !important;
    background: #ffffff !important;
    padding: 14px 16px 16px 16px !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05) !important;
    border: 1px solid var(--line) !important;
    margin-bottom: 16px !important;
}

.schematic-title {
    margin: 0 0 12px 0 !important;
    color: #0f172a !important;
    font-size: 16px !important;
    font-weight: 740 !important;
}

.status-card {
    text-align: center !important;
}

table {
    margin-left: auto !important;
    margin-right: auto !important;
    text-align: center !important;
}

.gr-button {
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Compact plot blocks while retaining readability. */
.plot-container, .gradio-plot {
    border-radius: 10px !important;
}

/* Gradio file component: visually aligned with the original layout. */
.file-preview, .file-preview-holder {
    text-align: center !important;
}

@media (max-width: 1100px) {
    .gradio-container {
        padding-left: 12px !important;
        padding-right: 12px !important;
    }
    .center-title h1 {
        font-size: 24px !important;
    }
}
"""


# ============================================================
# 2. HTML helpers
# ============================================================
def var(symbol, sub=None):
    """Return italic HTML variable with optional subscript."""
    if sub is None:
        return f"<i>{symbol}</i>"
    return f"<i>{symbol}</i><sub>{sub}</sub>"


def make_field_label(label_html, note_html=None):
    html = f"<div class='field-label'>{label_html}</div>"
    if note_html:
        html += f"<div class='field-note'>{note_html}</div>"
    return html


def get_schematic_html():
    return f"""
    <div class="schematic-box">
        <h3 class="schematic-title">Center-fed counter-current fractional extraction cascade</h3>

        <div style="display:flex; justify-content:center; align-items:stretch; gap:14px; width:100%; margin:0 auto;">

            <div style="flex:1; background:linear-gradient(180deg,#eff6ff,#dbeafe); border:1px solid #bfdbfe; border-radius:10px; padding:15px; min-height:112px;">
                <div style="font-weight:740; color:#1d4ed8; font-size:15.5px;">Scrubbing section</div>
                <div style="margin-top:10px; line-height:1.65; color:#1e293b; font-size:13.5px;">
                    Fresh scrubbing phase: {var("W", "vol")}<br>
                    Loaded organic outlet: x-enriched product
                </div>
            </div>

            <div style="width:220px; background:linear-gradient(180deg,#fff7ed,#ffedd5); border:1px solid #fed7aa; border-radius:10px; padding:15px; min-height:112px;">
                <div style="font-weight:740; color:#c2410c; font-size:15.5px;">Feed stage</div>
                <div style="margin-top:10px; line-height:1.65; color:#1e293b; font-size:13.5px;">
                    Stage {var("M")}<br>
                    Feed stream: {var("F", "vol")}
                </div>
            </div>

            <div style="flex:1; background:linear-gradient(180deg,#ecfdf5,#dcfce7); border:1px solid #bbf7d0; border-radius:10px; padding:15px; min-height:112px;">
                <div style="font-weight:740; color:#15803d; font-size:15.5px;">Extraction section</div>
                <div style="margin-top:10px; line-height:1.65; color:#1e293b; font-size:13.5px;">
                    Fresh organic phase: {var("S", "vol")}<br>
                    Aqueous raffinate outlet: y-enriched product
                </div>
            </div>

        </div>

        <div style="margin-top:11px; color:#64748b; font-size:12.3px;">
            Convention: isotope x is the preferentially extracted isotope, whereas isotope y is the less extractable isotope.
        </div>
    </div>
    """


# ============================================================
# 3. Numerical utilities
# ============================================================
def thomas_solve(lower, diag, upper, rhs):
    """
    Solve a tridiagonal linear system using the Thomas algorithm.
    """
    n = len(diag)
    a = np.array(lower, dtype=float).copy()
    b = np.array(diag, dtype=float).copy()
    c = np.array(upper, dtype=float).copy()
    d = np.array(rhs, dtype=float).copy()

    eps = 1e-30

    for i in range(1, n):
        m = a[i - 1] / (b[i - 1] + eps)
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    x = np.zeros(n)
    x[-1] = d[-1] / (b[-1] + eps)

    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / (b[i] + eps)

    return x


def build_tridiagonal_system(W, S, D, feed_index, feed_flux):
    """
    Build the tridiagonal steady-state balance equations for a center-fed counter-current cascade.
    """
    n = len(W)
    lower = -W[:-1]
    diag = W + S * D
    upper = -S[1:] * D[1:]
    rhs = np.zeros(n)
    rhs[feed_index] = feed_flux
    return lower, diag, upper, rhs


def compute_window(W, F, S, beta, D_scrub, D_ext):
    """
    Hydrodynamic operating-window criterion.

    Effective operation requires:
        S_vol > (W_vol + F_vol)/(beta*D_ext)
        S_vol < W_vol/D_scrub
    """
    W = float(W)
    F = float(F)
    S = float(S)
    beta = float(beta)
    D_scrub = float(D_scrub)
    D_ext = float(D_ext)

    if W <= 0 or F < 0 or S <= 0 or beta <= 1.0 or D_scrub <= 0 or D_ext <= 0:
        raise ValueError("Use W_vol > 0, F_vol ≥ 0, S_vol > 0, β > 1, and positive distribution ratios.")

    S_min = (W + F) / (beta * D_ext)
    S_max = W / D_scrub
    width = S_max - S_min

    E_x_ext = (S * beta * D_ext) / (W + F)
    E_y_scrub = (S * D_scrub) / W

    if width <= 0:
        regime = "No feasible operating window"
    elif S < S_min:
        regime = "Total Stripping region"
    elif S > S_max:
        regime = "Total Extraction region"
    else:
        regime = "Effective operating window"

    return S_min, S_max, width, E_x_ext, E_y_scrub, regime


# ============================================================
# 4. Operability-window display and map
# ============================================================
def make_operability_map(W, F, S, beta, D_scrub, D_ext):
    fig, ax = plt.subplots(figsize=(8.2, 3.75))

    try:
        W = float(W)
        F = float(F)
        S = float(S)
        beta = float(beta)
        D_scrub = float(D_scrub)
        D_ext = float(D_ext)

        S_min, S_max, width, _, _, _ = compute_window(W, F, S, beta, D_scrub, D_ext)

        F_crit = beta * D_ext * W / D_scrub - W

        if F_crit > 0:
            F_upper = max(2.5 * F, 1.15 * F_crit, 0.1)
        else:
            F_upper = max(2.5 * F, W, 0.1)

        S_upper = max(1.6 * S, 1.35 * S_max, 0.1)

        F_grid = np.linspace(0, F_upper, 260)
        S_grid = np.linspace(0, S_upper, 260)
        F_mesh, S_mesh = np.meshgrid(F_grid, S_grid)

        S_min_grid = (W + F_mesh) / (beta * D_ext)
        S_max_grid = W / D_scrub

        region = np.where(
            S_mesh < S_min_grid,
            0,
            np.where(S_mesh > S_max_grid, 2, 1)
        )

        cmap = ListedColormap(["#fee2e2", "#dcfce7", "#ede9fe"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        ax.contourf(F_mesh, S_mesh, region, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, norm=norm, alpha=0.92)

        S_min_line = (W + F_grid) / (beta * D_ext)
        ax.plot(F_grid, S_min_line, color="#b91c1c", linewidth=2.0, label=r"$S_{\min}$")
        ax.axhline(S_max, color="#6d28d9", linewidth=2.0, label=r"$S_{\max}$")

        ax.scatter([F], [S], color="#0f172a", edgecolor="white", linewidth=0.8, s=58, zorder=5, label="Current point")
        ax.annotate(
            "Current point",
            xy=(F, S),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8.5,
            ha="left",
            color="#0f172a"
        )

        ax.set_xlim(0, F_upper)
        ax.set_ylim(0, S_upper)
        ax.set_xlabel(r"$F_{\mathrm{vol}}$ (L h$^{-1}$)")
        ax.set_ylabel(r"$S_{\mathrm{vol}}$ (L h$^{-1}$)")
        ax.set_title(r"$F_{\mathrm{vol}}$–$S_{\mathrm{vol}}$ operability map", pad=8)

        legend_elements = [
            Patch(facecolor="#fee2e2", edgecolor="none", label="Stripping-dominant"),
            Patch(facecolor="#dcfce7", edgecolor="none", label="Effective window"),
            Patch(facecolor="#ede9fe", edgecolor="none", label="Extraction-dominant"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7.6, frameon=True, framealpha=0.92)

        ax.grid(True, alpha=0.22)
        for spine in ax.spines.values():
            spine.set_color("#94a3b8")
        plt.tight_layout()

    except Exception as e:
        ax.text(
            0.5, 0.5,
            f"Unable to generate operability map:\n{str(e)}",
            ha="center", va="center", fontsize=10.5, color="#991b1b"
        )
        ax.set_axis_off()

    return fig


def update_window_outputs(W, F, S, beta, D_scrub, D_ext):
    try:
        S_min, S_max, width, E_x_ext, E_y_scrub, regime = compute_window(W, F, S, beta, D_scrub, D_ext)

        W = float(W)
        F = float(F)
        S = float(S)

        if width <= 0:
            status_color = "#b91c1c"
            status_bg = "#fef2f2"
            status_text = (
                "No feasible window is available. Under the current distribution ratios and flow rates, "
                "the lower bound is not smaller than the upper bound."
            )
            bar_html = ""
            lower_margin = np.nan
            upper_margin = np.nan

        else:
            if S < S_min:
                status_color = "#b91c1c"
                status_bg = "#fef2f2"
                status_text = (
                    "Infeasible: the organic-phase flow rate is below the lower bound, "
                    "so x cannot be sufficiently transported into the organic phase in the extraction section."
                )
            elif S > S_max:
                status_color = "#6d28d9"
                status_bg = "#f5f3ff"
                status_text = (
                    "Infeasible: the organic-phase flow rate exceeds the upper bound, "
                    "so y is excessively extracted in the scrubbing section."
                )
            else:
                status_color = "#15803d"
                status_bg = "#ecfdf5"
                status_text = (
                    "Feasible: the current organic-phase flow rate lies inside the effective operating window."
                )

            lower_margin = (S - S_min) / S_min * 100 if S_min > 0 else np.nan
            upper_margin = (S_max - S) / S_max * 100 if S_max > 0 else np.nan

            display_min = min(S_min - 0.8 * width, S)
            display_max = max(S_max + 0.8 * width, S)
            display_range = max(display_max - display_min, 1e-12)

            left_pct = ((S_min - display_min) / display_range) * 100
            right_pct = ((S_max - display_min) / display_range) * 100
            s_pct = ((S - display_min) / display_range) * 100

            left_pct = np.clip(left_pct, 0, 100)
            right_pct = np.clip(right_pct, 0, 100)
            s_pct = np.clip(s_pct, 0, 100)

            bar_html = f"""
            <div style="margin:40px auto 72px auto; position:relative; height:50px;
                        background:#e2e8f0; border-radius:9px; overflow:visible; width:96%; border:1px solid #dbe3ef;">

                <div style="position:absolute; left:0; top:0; width:{left_pct:.2f}%; height:100%;
                            background:#fecaca; border-radius:9px 0 0 9px;
                            display:flex; align-items:center; justify-content:center;">
                    <span style="font-size:11.5px; color:#991b1b; font-weight:740;">Stripping-dominant</span>
                </div>

                <div style="position:absolute; left:{left_pct:.2f}%; top:0;
                            width:{max(right_pct - left_pct, 0):.2f}%; height:100%;
                            background:#bbf7d0; border-left:2px dashed #16a34a;
                            border-right:2px dashed #16a34a;
                            display:flex; align-items:center; justify-content:center;">
                    <span style="font-size:11.5px; color:#166534; font-weight:740;">
                        Effective operating window
                    </span>
                </div>

                <div style="position:absolute; left:{right_pct:.2f}%; top:0;
                            width:{max(100 - right_pct, 0):.2f}%; height:100%;
                            background:#ddd6fe; border-radius:0 9px 9px 0;
                            display:flex; align-items:center; justify-content:center;">
                    <span style="font-size:11.5px; color:#5b21b6; font-weight:740;">Extraction-dominant</span>
                </div>

                <div style="position:absolute; left:{s_pct:.2f}%; top:-16px;
                            width:4px; height:82px; background:#0f172a;
                            transform:translateX(-50%); box-shadow:0 0 5px rgba(15,23,42,0.35);">
                </div>

                <div style="position:absolute; left:{s_pct:.2f}%; top:-42px;
                            transform:translateX(-50%); background:#ffffff;
                            border:2px solid #0f172a; padding:3px 10px;
                            border-radius:14px; font-weight:740; font-size:12.5px;
                            color:#0f172a; white-space:nowrap;">
                    Current {var("S", "vol")} = {S:.4f}
                </div>

                <div style="position:absolute; left:{left_pct:.2f}%; top:60px;
                            transform:translateX(-50%); font-size:12.5px;
                            color:#334155; font-weight:740; white-space:nowrap;">
                    {var("S", "min")} = {S_min:.4f}
                </div>

                <div style="position:absolute; left:{right_pct:.2f}%; top:60px;
                            transform:translateX(-50%); font-size:12.5px;
                            color:#334155; font-weight:740; white-space:nowrap;">
                    {var("S", "max")} = {S_max:.4f}
                </div>
            </div>
            """

        margin_html = ""
        if width > 0:
            margin_html = f"""
            <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-top:8px;">
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; color:#334155; min-width:138px;">
                    Lower-bound margin<br>
                    <b>{lower_margin:.2f}%</b>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; color:#334155; min-width:138px;">
                    Upper-bound margin<br>
                    <b>{upper_margin:.2f}%</b>
                </div>
            </div>
            """

        html_content = f"""
        <div class="panel-card status-card">
            <h3 style="margin:0 0 8px 0; color:#0f172a; font-size:16px; font-weight:760;">Operating-window analyzer</h3>

            <div style="font-size:12.5px; color:#64748b; margin-bottom:10px; line-height:1.65;">
                Feasibility criterion for a center-fed cascade:
                {var("S", "min")} &lt; {var("S", "vol")} &lt; {var("S", "max")},
                where
                {var("S", "min")} = ({var("W", "vol")} + {var("F", "vol")}) /
                ({var("β")} · {var("D", "ext")})
                and
                {var("S", "max")} = {var("W", "vol")} / {var("D", "scrub")}.
            </div>

            <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-bottom:11px;">
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; min-width:105px;">
                    {var("S", "min")}<br><b>{S_min:.4f}</b>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; min-width:105px;">
                    {var("S", "max")}<br><b>{S_max:.4f}</b>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; min-width:118px;">
                    Window width<br><b>{width:.4f}</b>
                </div>
                <div style="background:{status_bg}; border:1px solid {status_color}; border-radius:8px; padding:9px 15px; min-width:190px;">
                    Predicted regime<br><b style="color:{status_color};">{regime}</b>
                </div>
            </div>

            <div style="background:{status_bg}; color:{status_color}; border:1px solid {status_color};
                        border-radius:8px; padding:8px 12px; font-weight:720; margin:10px auto; max-width:1000px; font-size:12.8px;">
                {status_text}
            </div>

            {bar_html}

            <div style="display:flex; justify-content:center; gap:16px; flex-wrap:wrap; margin-top:10px;">
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:11px 16px; min-width:285px;">
                    Extraction-section transport factor for isotope x<br>
                    {var("E", "x,ext")} &gt; 1 required<br>
                    <b style="font-size:19px; color:{'#15803d' if E_x_ext > 1 else '#b91c1c'};">{E_x_ext:.4f}</b>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:11px 16px; min-width:285px;">
                    Scrubbing-section transport factor for isotope y<br>
                    {var("E", "y,scrub")} &lt; 1 required<br>
                    <b style="font-size:19px; color:{'#15803d' if E_y_scrub < 1 else '#b91c1c'};">{E_y_scrub:.4f}</b>
                </div>
            </div>

            {margin_html}
        </div>
        """

        fig_map = make_operability_map(W, F, S, beta, D_scrub, D_ext)
        return html_content, fig_map

    except Exception as e:
        error_html = f"""
        <div class="panel-card status-card">
            <h3 style="margin-top:0; color:#991b1b;">Operating-window analyzer</h3>
            <div style="background:#fef2f2; color:#991b1b; border:1px solid #ef4444;
                        border-radius:8px; padding:12px; font-weight:720;">
                Invalid input: {str(e)}
            </div>
        </div>
        """
        fig = make_operability_map(1.0, 0.1, 0.8, 1.02, 0.5, 1.6)
        return error_html, fig


# ============================================================
# 5. Core TDMA solver with capacity-limited equilibrium
# ============================================================
def solve_fractional_extraction(
    N, M, W_vol, F_vol, S_vol, F_conc, alpha_feed, beta, D_scrub, D_ext, C_max
):
    try:
        N = int(N)
        M = int(M)
        W_vol = float(W_vol)
        F_vol = float(F_vol)
        S_vol = float(S_vol)
        F_conc = float(F_conc)
        alpha_feed = float(alpha_feed)
        beta = float(beta)
        D_scrub = float(D_scrub)
        D_ext = float(D_ext)
        C_max_val = float(C_max)

        if C_max_val <= 0:
            raise ValueError("Extractant saturation capacity must be positive.")
        if N < 2:
            raise ValueError("The total number of theoretical stages must be at least 2.")
        if not (1 <= M <= N):
            raise ValueError("The feed stage must be between 1 and N.")
        if W_vol <= 0 or F_vol < 0 or S_vol <= 0:
            raise ValueError("Use W_vol > 0, F_vol ≥ 0, and S_vol > 0.")
        if F_conc <= 0 or alpha_feed <= 0:
            raise ValueError("Feed concentration and feed isotope ratio must be positive.")
        if beta <= 1.0 or D_scrub <= 0 or D_ext <= 0:
            raise ValueError("Use β > 1 and positive distribution ratios.")

        feed_index = M - 1
        stages = np.arange(1, N + 1)

        # Baseline total distribution ratio before capacity correction.
        W = np.zeros(N)
        S = np.full(N, S_vol)
        D_ideal = np.zeros(N)

        for i in range(N):
            if i < feed_index:
                W[i] = W_vol
                D_ideal[i] = D_scrub
            else:
                W[i] = W_vol + F_vol
                D_ideal[i] = D_ext

        # ----------------------------------------------------
        # Picard iteration for the total concentration profile
        # with a hard organic-phase capacity limit.
        # ----------------------------------------------------
        eps = 1e-30

        lower, diag, upper, rhs = build_tridiagonal_system(W, S, D_ideal, feed_index, F_vol * F_conc)
        A_total = thomas_solve(lower, diag, upper, rhs)

        damping_sat = 0.5
        sat_errors = []
        for _ in range(3000):
            D_eff = np.zeros(N)
            for i in range(N):
                O_theoretical = D_ideal[i] * A_total[i]
                if O_theoretical > C_max_val:
                    D_eff[i] = C_max_val / (A_total[i] + eps)
                else:
                    D_eff[i] = D_ideal[i]

            lower, diag, upper, rhs = build_tridiagonal_system(W, S, D_eff, feed_index, F_vol * F_conc)
            A_new = thomas_solve(lower, diag, upper, rhs)

            err = np.max(np.abs(A_new - A_total) / (np.abs(A_total) + eps))
            sat_errors.append(err)
            A_total = damping_sat * A_new + (1.0 - damping_sat) * A_total

            if err < 1e-12:
                break

        # Recompute final D_eff and O_total, ensuring O_i <= C_max.
        D_eff = np.zeros(N)
        for i in range(N):
            if D_ideal[i] * A_total[i] > C_max_val:
                D_eff[i] = C_max_val / (A_total[i] + eps)
            else:
                D_eff[i] = D_ideal[i]
        O_total = D_eff * A_total

        # ----------------------------------------------------
        # Isotope resolution loop
        # ----------------------------------------------------
        z_x = F_conc * alpha_feed / (1.0 + alpha_feed)
        z_y = F_conc - z_x
        a = A_total * alpha_feed / (1.0 + alpha_feed)

        tolerance = 1e-12
        max_iter = 3000
        damping = 0.5
        errors_history = []

        for _ in range(max_iter):
            D_x = (beta * O_total) / (A_total + a * (beta - 1.0) + eps)

            lower_x, diag_x, upper_x, rhs_x = build_tridiagonal_system(
                W=W, S=S, D=D_x, feed_index=feed_index, feed_flux=F_vol * z_x
            )

            a_pred = thomas_solve(lower_x, diag_x, upper_x, rhs_x)

            rel_error = np.max(np.abs(a_pred - a) / (np.abs(a) + eps))
            errors_history.append(rel_error)

            a = damping * a_pred + (1.0 - damping) * a

            if rel_error < tolerance:
                break

        # Final profiles
        D_x = (beta * O_total) / (A_total + a * (beta - 1.0) + eps)
        D_y = D_x / beta

        b = D_x * a
        c = A_total - a
        d = O_total - b

        # Product purity / abundance
        x_abundance_org = b / (b + d + eps)
        y_abundance_aq = c / (a + c + eps)

        # Product recovery
        x_recovery_org = (S_vol * b) / (F_vol * z_x + eps)
        y_recovery_aq = ((W_vol + F_vol) * c) / (F_vol * z_y + eps)

        # Local extraction factors
        E_x = S * D_x / (W + eps)
        E_y = S * D_y / (W + eps)

        # Stage-wise residuals
        mb_x, mb_y = [], []
        for i in range(N):
            in_x = (
                (W[i - 1] * a[i - 1] if i > 0 else 0)
                + (S[i + 1] * b[i + 1] if i < N - 1 else 0)
                + (F_vol * z_x if i == feed_index else 0)
            )
            in_y = (
                (W[i - 1] * c[i - 1] if i > 0 else 0)
                + (S[i + 1] * d[i + 1] if i < N - 1 else 0)
                + (F_vol * z_y if i == feed_index else 0)
            )

            out_x = W[i] * a[i] + S[i] * b[i]
            out_y = W[i] * c[i] + S[i] * d[i]

            mb_x.append(abs(in_x - out_x) / (abs(in_x) + eps))
            mb_y.append(abs(in_y - out_y) / (abs(in_y) + eps))

        mb_x, mb_y = np.array(mb_x), np.array(mb_y)
        eq_x = np.abs(b - D_x * a) / (np.abs(b) + np.abs(D_x * a) + eps)
        eq_y = np.abs(d - D_y * c) / (np.abs(d) + np.abs(D_y * c) + eps)
        beta_residual = np.abs((D_x / (D_y + eps)) - beta) / (beta + eps)
        
        # ====================================================
        # APPLY MACHINE EPSILON TRUNCATION FOR ZERO-RESIDUALS
        # (Affects both the plots and the exported Excel data)
        # ====================================================
        mb_x = np.maximum(mb_x, 1e-16)
        mb_y = np.maximum(mb_y, 1e-16)
        eq_x = np.maximum(eq_x, 1e-16)
        eq_y = np.maximum(eq_y, 1e-16)
        beta_residual = np.maximum(beta_residual, 1e-16)
        errors_history = np.maximum(np.array(errors_history), 1e-16)

        # Global outlet mass closure
        feed_x_flux = F_vol * z_x
        feed_y_flux = F_vol * z_y
        outlet_x_flux = S_vol * b[0] + (W_vol + F_vol) * a[-1]
        outlet_y_flux = S_vol * d[0] + (W_vol + F_vol) * c[-1]
        closure_x = abs(outlet_x_flux - feed_x_flux) / (abs(feed_x_flux) + eps)
        closure_y = abs(outlet_y_flux - feed_y_flux) / (abs(feed_y_flux) + eps)

        # Operating window
        S_min, S_max, width, E_x_ext, E_y_scrub, regime = compute_window(
            W_vol, F_vol, S_vol, beta, D_scrub, D_ext
        )

        # ====================================================
        # Capacity diagnostics HTML
        # ====================================================
        peak_O = np.max(O_total)
        is_saturated = (peak_O >= C_max_val * 0.999)
        cap_text = f"{C_max_val:.4g}"

        sat_color = "#c2410c" if is_saturated else "#15803d"
        sat_bg = "#fff7ed" if is_saturated else "#ecfdf5"
        sat_border = "#fed7aa" if is_saturated else "#bbf7d0"
        sat_icon = "⚠️ Capacity limit active" if is_saturated else "✅ Capacity limit inactive"

        saturation_html = f"""
        <div style="margin-top:12px; padding:8px 12px; border-radius:8px; background:{sat_bg}; border:1px solid {sat_border}; color:{sat_color}; font-size:12.8px; text-align:center;">
            <b style="font-size:13.5px;">{sat_icon}</b>: peak organic-phase concentration is <b>{peak_O:.4e}</b> mol L<sup>−1</sup>
            (capacity limit: <b>{cap_text}</b> mol L<sup>−1</sup>).
        </div>
        """

        closure_color = "#15803d" if max(closure_x, closure_y) < 1e-8 else "#c2410c"
        closure_bg = "#ecfdf5" if max(closure_x, closure_y) < 1e-8 else "#fff7ed"
        closure_html = f"""
        <div style="margin-top:10px; padding:8px 12px; border-radius:8px; background:{closure_bg}; border:1px solid #e2e8f0; color:{closure_color}; font-size:12.8px; text-align:center;">
            Global outlet mass closure: isotope x = <b>{closure_x:.2e}</b>, isotope y = <b>{closure_y:.2e}</b>.
        </div>
        """

        # ====================================================
        # HTML summary
        # ====================================================
        summary_html = f"""
        <div class="panel-card">
            <h3 style="margin:0 0 10px 0; color:#0f172a; font-size:16px; font-weight:760;">Separation performance summary</h3>

            <div style="margin:0 auto 15px auto; color:#334155; line-height:1.7; font-size:12.8px;">
                Feed composition:
                {var("C", "F")} = <b>{F_conc:.4g}</b> mol L<sup>−1</sup>;
                {var("C", "x,F")} = <b>{z_x:.4e}</b> mol L<sup>−1</sup>;
                {var("C", "y,F")} = <b>{z_y:.4e}</b> mol L<sup>−1</sup>;
                {var("α", "F")} = {var("C", "x,F")}/{var("C", "y,F")} = <b>{alpha_feed:.4g}</b>.
            </div>

            <div style="display:flex; justify-content:center; gap:18px; flex-wrap:wrap;">

                <div style="flex:1; min-width:320px; max-width:520px; background:#ffffff;
                            border:1px solid #fecaca; border-top:4px solid #dc2626;
                            border-radius:9px; padding:15px; text-align:center;">
                    <h4 style="margin:0 0 8px 0; color:#b91c1c; font-size:14px;">Loaded organic outlet, Stage 1</h4>
                    <div style="line-height:1.9; color:#1e293b; font-size:12.8px;">
                        Isotope x concentration:
                        <b>{b[0]:.4e}</b> mol L<sup>−1</sup><br>
                        Isotope x abundance:
                        <b style="color:#b91c1c;">{x_abundance_org[0] * 100:.4f}%</b><br>
                        Isotope x recovery:
                        <b style="color:#b91c1c;">{x_recovery_org[0] * 100:.4f}%</b>
                    </div>
                </div>

                <div style="flex:1; min-width:320px; max-width:520px; background:#ffffff;
                            border:1px solid #bfdbfe; border-top:4px solid #2563eb;
                            border-radius:9px; padding:15px; text-align:center;">
                    <h4 style="margin:0 0 8px 0; color:#1d4ed8; font-size:14px;">Aqueous raffinate outlet, Stage N</h4>
                    <div style="line-height:1.9; color:#1e293b; font-size:12.8px;">
                        Isotope y concentration:
                        <b>{c[-1]:.4e}</b> mol L<sup>−1</sup><br>
                        Isotope y abundance:
                        <b style="color:#1d4ed8;">{y_abundance_aq[-1] * 100:.4f}%</b><br>
                        Isotope y recovery:
                        <b style="color:#1d4ed8;">{y_recovery_aq[-1] * 100:.4f}%</b>
                    </div>
                </div>

            </div>

            <div style="margin-top:15px; color:#64748b; font-size:12.5px; line-height:1.7;">
                Predicted operating regime:
                <b>{regime}</b>.
                Calculated bounds:
                {var("S", "min")} = <b>{S_min:.4f}</b>,
                {var("S", "max")} = <b>{S_max:.4f}</b>.
            </div>
            {saturation_html}
            {closure_html}
        </div>
        """

        # ====================================================
        # Figure 1: cascade profiles
        # ====================================================
        fig_prof, ax_prof = plt.subplots(1, 3, figsize=(14.8, 3.75))

        ax_prof[0].plot(stages, x_abundance_org, linewidth=2.0, label=r"Organic-phase x abundance")
        ax_prof[0].plot(stages, y_abundance_aq, linewidth=2.0, label=r"Aqueous-phase y abundance")
        ax_prof[0].axvline(M, linestyle="--", linewidth=1.1, color="#64748b")
        ax_prof[0].set_xlabel("Stage number")
        ax_prof[0].set_ylabel("Isotope abundance")
        ax_prof[0].set_title("Isotope abundance profiles")
        ax_prof[0].legend(fontsize=7.5)
        ax_prof[0].grid(True, alpha=0.25)
        ax_prof[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        ax_prof[1].plot(stages, a, linewidth=1.8, label=r"Aqueous x, $a_i$")
        ax_prof[1].plot(stages, b, linewidth=1.8, label=r"Organic x, $b_i$")
        ax_prof[1].plot(stages, c, linewidth=1.8, label=r"Aqueous y, $c_i$")
        ax_prof[1].plot(stages, d, linewidth=1.8, label=r"Organic y, $d_i$")
        ax_prof[1].axvline(M, linestyle="--", linewidth=1.1, color="#64748b")
        ax_prof[1].set_xlabel("Stage number")
        ax_prof[1].set_ylabel(r"Concentration (mol L$^{-1}$)")
        ax_prof[1].set_title("Absolute concentration profiles")
        ax_prof[1].legend(fontsize=7.2)
        ax_prof[1].grid(True, alpha=0.25)
        ax_prof[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        ax_prof[2].plot(stages, E_x, linewidth=2.0, label=r"$E_{x,i}=S_iD_{x,i}/W_i$")
        ax_prof[2].plot(stages, E_y, linewidth=2.0, label=r"$E_{y,i}=S_iD_{y,i}/W_i$")
        ax_prof[2].axhline(1.0, linestyle="--", linewidth=1.1, color="#64748b", label=r"$E=1$")
        ax_prof[2].axvline(M, linestyle="--", linewidth=1.1, color="#64748b")
        ax_prof[2].set_xlabel("Stage number")
        ax_prof[2].set_ylabel("Local extraction factor")
        ax_prof[2].set_title("Local transport criterion")
        ax_prof[2].legend(fontsize=7.2)
        ax_prof[2].grid(True, alpha=0.25)
        ax_prof[2].xaxis.set_major_locator(MaxNLocator(integer=True))

        for ax in ax_prof:
            for spine in ax.spines.values():
                spine.set_color("#94a3b8")
        plt.tight_layout()

        # ====================================================
        # Figure 2: numerical diagnostics
        # ====================================================
        fig_val, ax_val = plt.subplots(1, 3, figsize=(14.8, 3.75))

        ax_val[0].semilogy(np.arange(1, len(errors_history) + 1), errors_history, linewidth=1.8)
        ax_val[0].set_xlabel("Iteration number")
        ax_val[0].set_ylabel("Maximum relative update")
        ax_val[0].set_title("Convergence history")
        ax_val[0].grid(True, alpha=0.25)
        ax_val[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        ax_val[1].plot(stages, np.log10(mb_x), linewidth=1.8, label="Isotope x")
        ax_val[1].plot(stages, np.log10(mb_y), linewidth=1.8, label="Isotope y")
        ax_val[1].axvline(M, linestyle="--", linewidth=1.1, color="#64748b")
        ax_val[1].set_xlabel("Stage number")
        ax_val[1].set_ylabel(r"$\log_{10}$(relative residual)")
        ax_val[1].set_title("Stage-wise mass-balance residual")
        ax_val[1].legend(fontsize=7.5)
        ax_val[1].grid(True, alpha=0.25)
        ax_val[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        ax_val[2].plot(stages, np.log10(eq_x), linewidth=1.8, label=r"$b_i-D_{x,i}a_i$")
        ax_val[2].plot(stages, np.log10(eq_y), linewidth=1.8, label=r"$d_i-D_{y,i}c_i$")
        ax_val[2].plot(stages, np.log10(beta_residual), linewidth=1.5, label=r"$D_{x,i}/D_{y,i}-\beta$")
        ax_val[2].axvline(M, linestyle="--", linewidth=1.1, color="#64748b")
        ax_val[2].set_xlabel("Stage number")
        ax_val[2].set_ylabel(r"$\log_{10}$(relative residual)")
        ax_val[2].set_title("Thermodynamic consistency residuals")
        ax_val[2].legend(fontsize=6.8)
        ax_val[2].grid(True, alpha=0.25)
        ax_val[2].xaxis.set_major_locator(MaxNLocator(integer=True))

        for ax in ax_val:
            for spine in ax.spines.values():
                spine.set_color("#94a3b8")
        plt.tight_layout()

        # ====================================================
        # Excel output
        # ====================================================
        stage_df = pd.DataFrame({
            "Stage": stages,
            "Aqueous x, a_i (mol L^-1)": a,
            "Organic x, b_i (mol L^-1)": b,
            "Aqueous y, c_i (mol L^-1)": c,
            "Organic y, d_i (mol L^-1)": d,
            "Total Aqueous, A_i (mol L^-1)": A_total,
            "Total Organic, O_i (mol L^-1)": O_total,
            "Effective D_eff,i": D_eff,
            "D_x,i": D_x,
            "D_y,i": D_y,
            "E_x,i": E_x,
            "E_y,i": E_y,
            "Organic-phase x abundance": x_abundance_org,
            "Aqueous-phase y abundance": y_abundance_aq,
            "Mass-balance residual, isotope x": mb_x,
            "Mass-balance residual, isotope y": mb_y,
            "Equilibrium residual, isotope x": eq_x,
            "Equilibrium residual, isotope y": eq_y,
            "Separation-factor residual": beta_residual,
        })

        input_df = pd.DataFrame({
            "Parameter": [
                "Total number of theoretical stages, N",
                "Feed stage, M",
                "Total feed concentration, C_F",
                "Initial isotope ratio, alpha_F",
                "Extractant saturation capacity, C_max",
                "Single-stage separation factor, beta",
                "Distribution ratio in scrubbing section, D_scrub",
                "Distribution ratio in extraction section, D_ext",
                "Scrubbing-phase flow rate, W_vol",
                "Feed flow rate, F_vol",
                "Organic-phase flow rate, S_vol",
            ],
            "Value": [
                N, M, F_conc, alpha_feed, C_max_val,
                beta, D_scrub, D_ext, W_vol, F_vol, S_vol
            ]
        })

        window_df = pd.DataFrame({
            "Metric": [
                "S_min",
                "S_max",
                "Window width",
                "Extraction-section transport factor for isotope x",
                "Scrubbing-section transport factor for isotope y",
                "Predicted operating regime",
                "Global outlet closure, isotope x",
                "Global outlet closure, isotope y",
                "Number of isotope iterations",
                "Final maximum relative update",
                "Number of saturation iterations",
                "Final saturation relative update",
            ],
            "Value": [
                S_min,
                S_max,
                width,
                E_x_ext,
                E_y_scrub,
                regime,
                closure_x,
                closure_y,
                len(errors_history),
                errors_history[-1] if len(errors_history) > 0 else np.nan,
                len(sat_errors),
                sat_errors[-1] if len(sat_errors) > 0 else np.nan,
            ]
        })

        iteration_df = pd.DataFrame({
            "Iteration Number": np.arange(1, len(errors_history) + 1),
            "Max Relative Update": errors_history
        })

        temp_path = os.path.join(tempfile.gettempdir(), f"fractional_extraction_cascade_N{N}.xlsx")

        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            input_df.to_excel(writer, sheet_name="Inputs", index=False)
            window_df.to_excel(writer, sheet_name="Operating window", index=False)
            stage_df.to_excel(writer, sheet_name="Stagewise profiles", index=False)
            iteration_df.to_excel(writer, sheet_name="Convergence History", index=False)

        return summary_html, fig_prof, fig_val, temp_path

    except Exception as e:
        error_html = f"""
        <div class="panel-card">
            <h3 style="margin-top:0; color:#991b1b;">Simulation failed</h3>
            <div style="background:#fef2f2; color:#991b1b; border:1px solid #ef4444;
                        border-radius:8px; padding:12px; font-weight:720;">
                {str(e)}
            </div>
        </div>
        """
        return error_html, None, None, None


# ============================================================
# 6. Gradio interface
# ============================================================
def clamp_m_value(n, m):
    try:
        n = int(n)
        m = int(m)
        if m > n:
            return gr.update(value=n)
        if m < 1:
            return gr.update(value=1)
        return gr.update()
    except Exception:
        return gr.update()


with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as demo:

    gr.HTML(f"""
    <div class="center-title">
        <h1>Fractional Extraction Cascade Operability Analyzer</h1>
        <p>TDMA-based steady-state simulation of a center-fed isotope-enrichment cascade under near-unity separation factors</p>
    </div>
    """)

    gr.HTML(get_schematic_html())

    with gr.Row():

        # ==========================
        # Left input panel
        # ==========================
        with gr.Column(scale=1, elem_classes="left-panel"):

            with gr.Group(elem_classes="panel-card"):
                gr.HTML('<div class="section-heading">1. Cascade configuration</div>')

                gr.HTML(make_field_label(f"Total number of theoretical stages, {var('N')}"))
                N_input = gr.Number(value=50, show_label=False, precision=0)

                gr.HTML(make_field_label(f"Feed stage, {var('M')}"))
                M_input = gr.Number(value=42, show_label=False, precision=0)

                gr.HTML(make_field_label(
                    f"Total feed concentration, {var('C', 'F')} (mol L<sup>−1</sup>)"
                ))
                F_conc_input = gr.Number(value=0.02, show_label=False)

                gr.HTML(make_field_label(
                    f"Feed isotope ratio, {var('α', 'F')} = {var('C', 'x,F')} / {var('C', 'y,F')}"
                ))
                alpha_input = gr.Number(value=1.0, show_label=False)

                gr.HTML(make_field_label(
                    f"Organic-phase capacity limit, {var('C', 'max')} (mol L<sup>−1</sup>)"
                ))
                C_max_input = gr.Number(value=1.0, show_label=False)

            with gr.Group(elem_classes="panel-card"):
                gr.HTML('<div class="section-heading">2. Thermodynamic parameters</div>')

                gr.HTML(make_field_label(
                    f"Single-stage separation factor, {var('β')} = {var('D', 'x')} / {var('D', 'y')}",
                    "Use β &gt; 1; isotope x is defined as the preferentially extracted isotope."
                ))
                beta_input = gr.Number(value=1.04, show_label=False)

                gr.HTML(make_field_label(
                    f"Distribution ratio in the scrubbing section, {var('D', 'scrub')}"
                ))
                D_scrub_input = gr.Number(value=0.5, show_label=False)

                gr.HTML(make_field_label(
                    f"Distribution ratio in the extraction section, {var('D', 'ext')}"
                ))
                D_ext_input = gr.Number(value=1.8, show_label=False)

                gr.HTML(f"""
                <div class="field-note" style="margin-top:12px;">
                    The model uses section-dependent total distribution ratios and resolves isotope-specific distributions from β.
                </div>
                """)

            with gr.Group(elem_classes="panel-card"):
                gr.HTML('<div class="section-heading">3. Hydrodynamic parameters</div>')

                gr.HTML(make_field_label(
                    f"Scrubbing-phase flow rate, {var('W', 'vol')} (L h<sup>−1</sup>)"
                ))
                W_vol_input = gr.Number(value=0.7, show_label=False)

                gr.HTML(make_field_label(
                    f"Feed flow rate, {var('F', 'vol')} (L h<sup>−1</sup>)"
                ))
                F_vol_input = gr.Number(value=0.2, show_label=False)

                gr.HTML(make_field_label(
                    f"Organic-phase flow rate, {var('S', 'vol')} (L h<sup>−1</sup>)"
                ))
                S_vol_input = gr.Number(value=1.25, show_label=False)

                run_btn = gr.Button("Run steady-state simulation", variant="primary", elem_classes="big-btn")

            with gr.Accordion("Model assumptions", open=False):
                gr.HTML("""
                <div class="panel-card" style="box-shadow:none; border:none; margin-bottom:0; background:#ffffff;">
                    <div style="line-height:1.9; color:#334155; font-size:12.8px;">
                        Steady-state counter-current operation<br>
                        Capacity-limited organic-phase loading, with O<sub>i</sub> ≤ C<sub>max</sub><br>
                        Constant phase volumes and negligible excess mixing volume<br>
                        Complete immiscibility of the aqueous and organic phases<br>
                        No entrainment, axial mixing, chemical degradation, or stage-efficiency loss
                    </div>
                </div>
                """)

        # ==========================
        # Right output panel
        # ==========================
        with gr.Column(scale=2):

            window_display = gr.HTML()

            map_plot = gr.Plot(
                label="Operability map"
            )

            res_summary = gr.HTML(
                """
                <div class="panel-card">
                    <h3 style="margin-top:0; color:#64748b; font-size:15px;">
                        Click “Run steady-state simulation” to generate the separation report.
                    </h3>
                </div>
                """
            )

            prof_plot = gr.Plot(
                label="Steady-state cascade profiles"
            )

            val_plot = gr.Plot(
                label="Numerical diagnostics"
            )

            file_dl = gr.File(
                label="Download Excel report"
            )

    # ========================================================
    # Authorship footer
    # ========================================================
    gr.HTML("""
    <div style="text-align: center; margin-top: 36px; padding-bottom: 16px; color: #64748b; font-size: 18px;">
        Developed by: Dr. Qi ZHAO (HKU), Dr. Wei SHEN (LUH), Prof. Kaimin SHIH (HKU), Prof. Zheng Li (CAS), Mr. Junyi Wang (HKU), Mr. Aung Thit Htun (HKU)<br>
        For inquiries or suggestions regarding this project, please feel free to contact zhaoqi22@hku.hk (Q. ZHAO) or kshih@hku.hk (K. SHIH).
    </div>
    """)

    # ========================================================
    # Event bindings
    # ========================================================
    N_input.change(
        fn=clamp_m_value,
        inputs=[N_input, M_input],
        outputs=[M_input]
    )

    dynamic_inputs = [
        W_vol_input,
        F_vol_input,
        S_vol_input,
        beta_input,
        D_scrub_input,
        D_ext_input
    ]

    for inp in dynamic_inputs:
        inp.change(
            fn=update_window_outputs,
            inputs=dynamic_inputs,
            outputs=[window_display, map_plot]
        )

    demo.load(
        fn=update_window_outputs,
        inputs=dynamic_inputs,
        outputs=[window_display, map_plot]
    )

    run_btn.click(
        fn=solve_fractional_extraction,
        inputs=[
            N_input,
            M_input,
            W_vol_input,
            F_vol_input,
            S_vol_input,
            F_conc_input,
            alpha_input,
            beta_input,
            D_scrub_input,
            D_ext_input,
            C_max_input
        ],
        outputs=[
            res_summary,
            prof_plot,
            val_plot,
            file_dl
        ]
    )


if __name__ == "__main__":
    demo.launch()
