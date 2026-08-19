import os
import tempfile
import numpy as np
import pandas as pd

# ============================================================
# 0. Use the thread-safe non-interactive Matplotlib backend for server deployment.
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import gradio as gr

try:
    import scipy.linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

MIN_STAGE_COUNT = 1
MAX_STAGE_COUNT = 500
DEFAULT_CLOSURE_TOL = 5e-3
EPS = 1e-30

# ============================================================
# 1. Global plotting style
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
# 2. Global CSS
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

.uncertainty-guide {
    text-align: left !important;
    font-size: 11.8px !important;
    color: #334155 !important;
    line-height: 1.55 !important;
    margin: 2px 0 10px 0 !important;
    padding: 8px 10px !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 8px !important;
    background: #eff6ff !important;
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

.plot-container, .gradio-plot {
    border-radius: 10px !important;
}

.file-preview, .file-preview-holder {
    text-align: center !important;
}

.app-footer {
    margin: 18px auto 0 auto !important;
    padding: 14px 18px !important;
    max-width: 1320px !important;
    border-top: 1px solid var(--line) !important;
    color: var(--muted) !important;
    font-size: 12.5px !important;
    line-height: 1.7 !important;
    text-align: center !important;
}

.app-footer b {
    color: #334155 !important;
    font-weight: 700 !important;
}

.app-footer a {
    color: var(--blue) !important;
    text-decoration: none !important;
    font-weight: 650 !important;
}

.app-footer a:hover {
    text-decoration: underline !important;
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
# 3. HTML helpers
# ============================================================
def var(symbol, sub=None):
    if sub is None:
        return f"<i>{symbol}</i>"
    return f"<i>{symbol}</i><sub>{sub}</sub>"

def make_field_label(label_html, note_html=None):
    html = f"<div class='field-label'>{label_html}</div>"
    if note_html:
        html += f"<div class='field-note'>{note_html}</div>"
    return html

def make_uncertainty_guide(extra_html=""):
    """Explain the paired mean +/- SD inputs concisely."""
    extra = f"<br>{extra_html}" if extra_html else ""
    return (
        "<div class='uncertainty-guide'>"
        "<b>Input format:</b> assumed mean &plusmn; standard deviation "
        "(SD, 1&sigma;). Inputs are sampled independently; SD = 0 keeps the "
        "input fixed."
        f"{extra}</div>"
    )

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
# 4. Numerical utilities
# ============================================================
def thomas_solve(lower, diag, upper, rhs):
    n = len(diag)
    if HAS_SCIPY:
        ab = np.zeros((3, n))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        x = scipy.linalg.solve_banded((1, 1), ab, rhs)
    else:
        A = np.zeros((n, n))
        np.fill_diagonal(A, diag)
        np.fill_diagonal(A[1:], lower)
        np.fill_diagonal(A[:, 1:], upper)
        x = np.linalg.solve(A, rhs)
    return np.maximum(x, 0.0)

def validate_common_inputs(N=None, M=None, W_vol=None, F_vol=None, S_vol=None,
                           F_conc=None, alpha_feed=None, beta=None,
                           D_Sc=None, D_Ex=None, C_max=None):
    if N is not None:
        N = int(N)
        if N < MIN_STAGE_COUNT or N > MAX_STAGE_COUNT:
            raise ValueError(f"N must be between {MIN_STAGE_COUNT} and {MAX_STAGE_COUNT}.")
    if M is not None:
        M = int(M)
        if N is not None and not (1 <= M <= N):
            raise ValueError("Feed stage M must satisfy 1 <= M <= N.")
    checks = [
        ("W_vol", W_vol, 0.0, False),
        ("F_vol", F_vol, 0.0, True),
        ("S_vol", S_vol, 0.0, False),
        ("F_conc", F_conc, 0.0, False),
        ("alpha_feed", alpha_feed, 0.0, False),
        ("beta", beta, 1.0, False),
        ("D_Sc", D_Sc, 0.0, False),
        ("D_Ex", D_Ex, 0.0, False),
        ("C_max", C_max, 0.0, False),
    ]
    for name, value, bound, allow_equal in checks:
        if value is None:
            continue
        value = float(value)
        if allow_equal:
            valid = value >= bound
        else:
            valid = value > bound
        if not valid:
            op = ">=" if allow_equal else ">"
            raise ValueError(f"{name} must be {op} {bound}.")

def build_tridiagonal_system(W, S, D, feed_index, feed_flux):
    n = len(W)
    lower = -W[:-1]
    diag = W + S * D
    upper = -S[1:] * D[1:]
    rhs = np.zeros(n)
    rhs[feed_index] = feed_flux
    return lower, diag, upper, rhs

def isotope_specific_distribution_from_total(D_total, beta, alpha_ref):
    """
    Convert a measurable bulk distribution ratio into isotope-specific
    distribution ratios using the local isotope ratio alpha = C_x/C_y.

    D_total = (D_x*C_x + D_y*C_y)/(C_x + C_y), and beta = D_x/D_y.
    Therefore:
        D_x = beta*(1+alpha)/(1+beta*alpha)*D_total
        D_y =       (1+alpha)/(1+beta*alpha)*D_total

    In the operating-window pre-screen, alpha_ref is taken as the feed
    isotope ratio. The full TDMA solver still recalculates stage-specific
    D_x,i and D_y,i from the solved concentration profiles.
    """
    D_total = float(D_total)
    beta = float(beta)
    alpha_ref = float(alpha_ref)
    denom = 1.0 + beta * alpha_ref
    D_x = beta * (1.0 + alpha_ref) / denom * D_total
    D_y = (1.0 + alpha_ref) / denom * D_total
    return D_x, D_y

def compute_window(W, F, S, alpha_ref, beta, D_Sc, D_Ex):
    if any(val is None for val in [W, F, S, alpha_ref, beta, D_Sc, D_Ex]):
        raise ValueError("Please provide all hydrodynamic, isotopic, and thermodynamic parameters.")
    W, F, S, alpha_ref, beta, D_Sc, D_Ex = map(float, [W, F, S, alpha_ref, beta, D_Sc, D_Ex])
    if W <= 0 or F < 0 or S <= 0 or alpha_ref <= 0 or beta <= 1.0 or D_Sc <= 0 or D_Ex <= 0:
        raise ValueError("Use W_vol > 0, F_vol ≥ 0, S_vol > 0, alpha_F > 0, β > 1, and positive distribution ratios.")

    # Strict isotope-specific operating-window boundaries.
    # Lower bound: isotope x must be extractively transported in the extraction section.
    # Upper bound: isotope y must not be excessively carried by the organic phase in the scrubbing section.
    D_x_Ex, _ = isotope_specific_distribution_from_total(D_Ex, beta, alpha_ref)
    _, D_y_Sc = isotope_specific_distribution_from_total(D_Sc, beta, alpha_ref)

    S_min = (W + F) / D_x_Ex
    S_max = W / D_y_Sc
    width = S_max - S_min

    E_x_ext = (S * D_x_Ex) / (W + F)
    E_y_scrub = (S * D_y_Sc) / W

    if width <= 0: regime = "No feasible operating window"
    elif S < S_min: regime = "Total Stripping region"
    elif S > S_max: regime = "Total Extraction region"
    else: regime = "Effective operating window"
    return S_min, S_max, width, E_x_ext, E_y_scrub, regime


UNCERTAINTY_SAMPLE_COUNT = 12000
UNCERTAINTY_RANDOM_SEED = 20260814
OPERABILITY_SCREEN_SAMPLE_COUNT = 1600
DESIGN_UNCERTAINTY_SAMPLE_COUNT_FAST = 240
DESIGN_UNCERTAINTY_SAMPLE_COUNT_THOROUGH = 480
UNCERTAINTY_INTERVAL = (0.025, 0.975)


def validate_measurement_uncertainties(
    beta_err, D_Sc_err, D_Ex_err,
    W_err=0.0, F_err=0.0, S_err=0.0,
):
    """Validate the non-negative input standard deviations (SDs)."""
    names = (
        "SD(beta)", "SD(D_Sc)", "SD(D_Ex)",
        "SD(W_vol)", "SD(F_vol)", "SD(S_vol)",
    )
    values = (beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err)
    validated = []

    for name, value in zip(names, values):
        if value is None:
            raise ValueError(f"Please provide {name}.")
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite value >= 0.")
        validated.append(value)

    return tuple(validated)


def has_nonzero_uncertainty(*values):
    return any(float(value or 0.0) > 0.0 for value in values)


def sample_physical_normal(rng, mean, std, lower_bound, sample_count, name):
    """
    Sample a normal measurement model subject to the physical lower bound.

    Rejection sampling avoids the artificial point mass that simple clipping
    would create at D = 0 or beta = 1.
    """
    mean, std = float(mean), float(std)
    if std == 0:
        return np.full(sample_count, mean, dtype=float)

    accepted = []
    accepted_count = 0
    attempts = 0

    while accepted_count < sample_count and attempts < 60:
        remaining = sample_count - accepted_count
        candidates = rng.normal(mean, std, size=max(2 * remaining, 1024))
        candidates = candidates[np.isfinite(candidates) & (candidates > lower_bound)]
        if candidates.size:
            take = candidates[:remaining]
            accepted.append(take)
            accepted_count += take.size
        attempts += 1

    if accepted_count < sample_count:
        raise ValueError(
            f"The entered SD for {name} leaves too little probability "
            f"inside its physical domain."
        )

    return np.concatenate(accepted)[:sample_count]


def sample_uncertain_operating_parameters(
    rng, W, F, S, beta, D_Sc, D_Ex,
    W_err, F_err, S_err, beta_err, D_Sc_err, D_Ex_err,
    sample_count,
):
    """Draw independent, physically admissible thermodynamic and flow samples."""
    return {
        "W": sample_physical_normal(rng, W, W_err, 0.0, sample_count, "W_vol"),
        "F": sample_physical_normal(rng, F, F_err, 0.0, sample_count, "F_vol"),
        "S": sample_physical_normal(rng, S, S_err, 0.0, sample_count, "S_vol"),
        "beta": sample_physical_normal(rng, beta, beta_err, 1.0, sample_count, "beta"),
        "D_Sc": sample_physical_normal(rng, D_Sc, D_Sc_err, 0.0, sample_count, "D_Sc"),
        "D_Ex": sample_physical_normal(rng, D_Ex, D_Ex_err, 0.0, sample_count, "D_Ex"),
    }


def propagate_operability_uncertainty(
    W, F, S, alpha_ref, beta, D_Sc, D_Ex,
    beta_err, D_Sc_err, D_Ex_err,
    W_err=0.0, F_err=0.0, S_err=0.0,
    sample_count=UNCERTAINTY_SAMPLE_COUNT,
    random_seed=UNCERTAINTY_RANDOM_SEED,
    return_samples=False,
):
    """
    Propagate measurement uncertainty through the nonlinear window equations.

    The paired inputs define independent normal models as assumed mean +/-
    standard deviation (SD, 1 sigma). Samples outside the physical domains are
    rejected.
    A fixed-seed parametric Monte Carlo calculation returns equal-tailed
    95% propagated uncertainty intervals. Distribution ratios and flow rates
    are constrained to positive values and the separation factor to beta > 1.
    """
    beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err = validate_measurement_uncertainties(
        beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err
    )
    W, F, S, alpha_ref, beta, D_Sc, D_Ex = map(
        float, [W, F, S, alpha_ref, beta, D_Sc, D_Ex]
    )

    rng = np.random.default_rng(random_seed)
    samples = sample_uncertain_operating_parameters(
        rng, W, F, S, beta, D_Sc, D_Ex,
        W_err, F_err, S_err, beta_err, D_Sc_err, D_Ex_err,
        int(sample_count),
    )

    common = (1.0 + alpha_ref) / (1.0 + samples["beta"] * alpha_ref)
    D_x_Ex_samples = samples["beta"] * common * samples["D_Ex"]
    D_y_Sc_samples = common * samples["D_Sc"]

    S_min_samples = (samples["W"] + samples["F"]) / D_x_Ex_samples
    S_max_samples = samples["W"] / D_y_Sc_samples
    F_crit_samples = samples["W"] * (D_x_Ex_samples / D_y_Sc_samples - 1.0)
    feasible_samples = (
        (S_min_samples < S_max_samples)
        & (samples["S"] > S_min_samples)
        & (samples["S"] < S_max_samples)
    )

    result = {
        "S_min": np.quantile(S_min_samples, UNCERTAINTY_INTERVAL),
        "S_max": np.quantile(S_max_samples, UNCERTAINTY_INTERVAL),
        "F_crit": np.quantile(F_crit_samples, UNCERTAINTY_INTERVAL),
        "W": np.quantile(samples["W"], UNCERTAINTY_INTERVAL),
        "F": np.quantile(samples["F"], UNCERTAINTY_INTERVAL),
        "S": np.quantile(samples["S"], UNCERTAINTY_INTERVAL),
        "feasibility_probability": float(np.mean(feasible_samples)),
        "sample_count": int(sample_count),
    }
    if return_samples:
        result["samples"] = samples
        result["D_x_Ex_samples"] = D_x_Ex_samples
        result["D_y_Sc_samples"] = D_y_Sc_samples
    return result

# ============================================================
# 5. Fast solver
# ============================================================
def ultra_fast_solve(N, M, W_vol, F_vol, S_vol, F_conc, alpha_feed, beta, D_Sc, D_Ex, C_max_val, target_x=0, target_y=0, is_fine=False):
    try:
        validate_common_inputs(
            N=N, M=M, W_vol=W_vol, F_vol=F_vol, S_vol=S_vol, F_conc=F_conc,
            alpha_feed=alpha_feed, beta=beta, D_Sc=D_Sc, D_Ex=D_Ex, C_max=C_max_val
        )
        N, M = int(N), int(M)
        W_vol, F_vol, S_vol = map(float, [W_vol, F_vol, S_vol])
        F_conc, alpha_feed, beta = map(float, [F_conc, alpha_feed, beta])
        D_Sc, D_Ex, C_max_val = map(float, [D_Sc, D_Ex, C_max_val])
        eps = 1e-30
        feed_index = M - 1
        W = np.zeros(N)
        W[:feed_index] = W_vol
        W[feed_index:] = W_vol + F_vol
        S = np.full(N, S_vol)
        D_ideal = np.zeros(N)
        D_ideal[:feed_index] = D_Sc
        D_ideal[feed_index:] = D_Ex

        lower_W = -W[:-1]
        rhs_W = np.zeros(N)
        rhs_W[feed_index] = F_vol * F_conc
        ab_W = np.zeros((3, N))
        ab_W[2, :-1] = lower_W
        
        ab_W[0, 1:] = -S[1:] * D_ideal[1:]
        ab_W[1, :] = W + S * D_ideal
        if HAS_SCIPY:
            A_total = scipy.linalg.solve_banded((1, 1), ab_W, rhs_W)
        else:
            A_total = thomas_solve(lower_W, ab_W[1,:], ab_W[0,1:], rhs_W)
        A_total = np.maximum(A_total, 0.0)

        max_iter_sat = 2500 if is_fine else 250
        for step in range(max_iter_sat):
            O_theoretical = D_ideal * A_total
            D_eff = np.where(O_theoretical > C_max_val, C_max_val / (A_total + eps), D_ideal)
            ab_W[0, 1:] = -S[1:] * D_eff[1:]
            ab_W[1, :] = W + S * D_eff
            
            if HAS_SCIPY:
                A_new = scipy.linalg.solve_banded((1, 1), ab_W, rhs_W)
            else:
                A_new = thomas_solve(lower_W, ab_W[1,:], ab_W[0,1:], rhs_W)
            A_new = np.maximum(A_new, 0.0)

            err = np.max(np.abs(A_new - A_total) / (np.abs(A_total) + eps))
            damping_sat = 0.005 if err > 1.0 else (0.02 if err > 0.1 else (0.05 if err > 0.01 else 0.15))
            A_total = damping_sat * A_new + (1.0 - damping_sat) * A_total
            if err < (1e-8 if is_fine else 1e-4): break
            
        O_total = np.where(D_ideal * A_total > C_max_val, C_max_val, D_ideal * A_total)
        
        z_x = F_conc * alpha_feed / (1.0 + alpha_feed)
        z_y = F_conc - z_x
        rhs_x = np.zeros(N)
        rhs_x[feed_index] = F_vol * z_x
        a = A_total * alpha_feed / (1.0 + alpha_feed)
        
        max_iter_iso = 3000 if is_fine else 250
        for step in range(max_iter_iso):
            D_x = (beta * O_total) / (A_total + a * (beta - 1.0) + eps)
            ab_W[0, 1:] = -S[1:] * D_x[1:]
            ab_W[1, :] = W + S * D_x
            
            if HAS_SCIPY:
                a_pred = scipy.linalg.solve_banded((1, 1), ab_W, rhs_x)
            else:
                a_pred = thomas_solve(lower_W, ab_W[1,:], ab_W[0,1:], rhs_x)
            a_pred = np.clip(a_pred, 0.0, A_total)

            rel_error = np.max(np.abs(a_pred - a) / (np.abs(a) + eps))
            damping_iso = 0.005 if rel_error > 1.0 else (0.02 if rel_error > 0.1 else (0.05 if rel_error > 0.01 else 0.15))
            a = damping_iso * a_pred + (1.0 - damping_iso) * a
            a = np.clip(a, 0.0, A_total)
            
            if not is_fine and step == 60:
                D_x_temp = (beta * O_total) / (A_total + a * (beta - 1.0) + eps)
                b_temp = np.clip(D_x_temp * a, 0.0, O_total)
                c_temp = np.maximum(A_total - a, 0.0)
                d_temp = np.maximum(O_total - b_temp, 0.0)
                x_pur_est = b_temp[0] / (b_temp[0] + d_temp[0] + eps) * 100
                y_pur_est = c_temp[-1] / (a[-1] + c_temp[-1] + eps) * 100
                if (target_x > 0 and x_pur_est < target_x - 15.0) or (target_y > 0 and y_pur_est < target_y - 15.0):
                    return 0, 0, 0, 0

            if rel_error < (1e-8 if is_fine else 1e-4): break

        D_x = (beta * O_total) / (A_total + a * (beta - 1.0) + eps)
        b = np.clip(D_x * a, 0.0, O_total)
        c = np.maximum(A_total - a, 0.0)
        d = np.maximum(O_total - b, 0.0)

        outlet_x_flux = S_vol * b[0] + (W_vol + F_vol) * a[-1]
        outlet_y_flux = S_vol * d[0] + (W_vol + F_vol) * c[-1]
        feed_x_flux = F_vol * z_x
        feed_y_flux = F_vol * z_y
        closure_x = abs(outlet_x_flux - feed_x_flux) / (abs(feed_x_flux) + eps)
        closure_y = abs(outlet_y_flux - feed_y_flux) / (abs(feed_y_flux) + eps)
        if is_fine and (closure_x > 0.005 or closure_y > 0.005):
            return 0, 0, 0, 0

        x_pur = b[0] / (b[0] + d[0] + eps) * 100
        y_pur = c[-1] / (a[-1] + c[-1] + eps) * 100
        x_rec = min((S_vol * b[0]) / (F_vol * z_x + eps) * 100, 100.0)
        y_rec = min(((W_vol + F_vol) * c[-1]) / (F_vol * z_y + eps) * 100, 100.0)

        return x_pur, y_pur, x_rec, y_rec
    except:
        return 0, 0, 0, 0


def propagate_design_performance_uncertainty(
    N, M, W_vol, F_vol, S_vol, F_conc, alpha_feed,
    beta, D_Sc, D_Ex, C_max,
    beta_err=0.0, D_Sc_err=0.0, D_Ex_err=0.0,
    W_err=0.0, F_err=0.0, S_err=0.0,
    req_x=False, req_y=False, target_x=0.0, target_y=0.0,
    sample_count=DESIGN_UNCERTAINTY_SAMPLE_COUNT_FAST,
    random_seed=UNCERTAINTY_RANDOM_SEED + 101,
):
    """
    Propagate thermodynamic and phase-flow uncertainty to outlet performance.

    Intervals are equal-tailed 95% Monte Carlo propagation intervals conditional
    on successful numerical solves. Failed solves remain failures when target-
    attainment and convergence probabilities are calculated.
    """
    beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err = validate_measurement_uncertainties(
        beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err
    )
    uncertain = has_nonzero_uncertainty(
        beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err
    )
    requested_samples = int(sample_count) if uncertain else 1
    rng = np.random.default_rng(random_seed)
    samples = sample_uncertain_operating_parameters(
        rng, W_vol, F_vol, S_vol, beta, D_Sc, D_Ex,
        W_err, F_err, S_err, beta_err, D_Sc_err, D_Ex_err,
        requested_samples,
    )

    metrics = {"x_pur": [], "y_pur": [], "x_rec": [], "y_rec": []}
    joint_target_successes = 0
    operability_successes = 0

    for i in range(requested_samples):
        W_i = float(samples["W"][i])
        F_i = float(samples["F"][i])
        S_i = float(samples["S"][i])
        beta_i = float(samples["beta"][i])
        D_Sc_i = float(samples["D_Sc"][i])
        D_Ex_i = float(samples["D_Ex"][i])

        try:
            S_min_i, S_max_i, _, _, _, _ = compute_window(
                W_i, F_i, S_i, alpha_feed, beta_i, D_Sc_i, D_Ex_i
            )
            if S_min_i < S_i < S_max_i:
                operability_successes += 1
        except Exception:
            pass

        values = ultra_fast_solve(
            N, M, W_i, F_i, S_i, F_conc, alpha_feed,
            beta_i, D_Sc_i, D_Ex_i, C_max,
            target_x=0.0, target_y=0.0, is_fine=True,
        )
        values = np.asarray(values, dtype=float)
        solve_succeeded = (
            values.shape == (4,)
            and np.all(np.isfinite(values))
            and values[0] > 0.0
            and values[1] > 0.0
            and values[2] >= 0.0
            and values[3] >= 0.0
        )
        if not solve_succeeded:
            continue

        x_pur_i, y_pur_i, x_rec_i, y_rec_i = map(float, values)
        metrics["x_pur"].append(x_pur_i)
        metrics["y_pur"].append(y_pur_i)
        metrics["x_rec"].append(x_rec_i)
        metrics["y_rec"].append(y_rec_i)

        pass_x = (x_pur_i >= float(target_x)) if req_x else True
        pass_y = (y_pur_i >= float(target_y)) if req_y else True
        if pass_x and pass_y:
            joint_target_successes += 1

    successful_samples = len(metrics["x_pur"])
    if successful_samples == 0:
        raise ValueError(
            "All uncertainty-propagation solves failed; reduce the entered SDs "
            "or move the design farther inside the operating window."
        )

    intervals = {
        name: np.quantile(np.asarray(values, dtype=float), UNCERTAINTY_INTERVAL)
        for name, values in metrics.items()
    }
    nominal = ultra_fast_solve(
        N, M, W_vol, F_vol, S_vol, F_conc, alpha_feed,
        beta, D_Sc, D_Ex, C_max, is_fine=True,
    )
    nominal = dict(zip(("x_pur", "y_pur", "x_rec", "y_rec"), map(float, nominal)))
    nominal_inside_intervals = all(
        float(intervals[name][0]) <= nominal[name] <= float(intervals[name][1])
        for name in intervals
    )

    requested_lower_bounds_pass = True
    target_inside_any_interval = False
    target_above_any_interval = False
    if req_x:
        requested_lower_bounds_pass &= float(intervals["x_pur"][0]) >= float(target_x)
        target_inside_any_interval |= float(intervals["x_pur"][0]) <= float(target_x) <= float(intervals["x_pur"][1])
        target_above_any_interval |= float(target_x) > float(intervals["x_pur"][1])
    if req_y:
        requested_lower_bounds_pass &= float(intervals["y_pur"][0]) >= float(target_y)
        target_inside_any_interval |= float(intervals["y_pur"][0]) <= float(target_y) <= float(intervals["y_pur"][1])
        target_above_any_interval |= float(target_y) > float(intervals["y_pur"][1])

    solve_probability = successful_samples / requested_samples
    joint_target_probability = joint_target_successes / requested_samples
    robust_target_met = bool(
        requested_lower_bounds_pass
        and solve_probability >= 0.95
        and (operability_successes / requested_samples) >= 0.95
        and joint_target_probability >= 0.95
    )

    return {
        "intervals": intervals,
        "nominal": nominal,
        "nominal_inside_intervals": bool(nominal_inside_intervals),
        "robust_target_met": robust_target_met,
        "target_inside_any_interval": bool(target_inside_any_interval),
        "target_above_any_interval": bool(target_above_any_interval),
        "joint_target_probability": float(joint_target_probability),
        "solve_probability": float(solve_probability),
        "operability_probability": float(operability_successes / requested_samples),
        "requested_samples": int(requested_samples),
        "successful_samples": int(successful_samples),
        "uncertainty_active": bool(uncertain),
    }


def make_uncertainty_status_html(report, req_x=False, req_y=False, target_x=0.0, target_y=0.0):
    """Create an explicit, claim-calibrated propagated-interval status card."""
    intervals = report["intervals"]
    x_ci = intervals["x_pur"]
    y_ci = intervals["y_pur"]
    xr_ci = intervals["x_rec"]
    yr_ci = intervals["y_rec"]

    if report["robust_target_met"] and (req_x or req_y):
        color, bg, border = "#166534", "#ecfdf5", "#22c55e"
        heading = "95% ROBUST TARGET SATISFACTION"
        statement = "The lower 95% interval bound meets every selected abundance target, and at least 95% of all requested Monte Carlo trials satisfy the joint target."
    elif report["target_inside_any_interval"] and (req_x or req_y):
        color, bg, border = "#92400e", "#fffbeb", "#f59e0b"
        heading = "TARGET LIES WITHIN THE 95% PROPAGATED INTERVAL"
        statement = "The nominal design reaches the target, but uncertainty spans the target boundary; this is uncertainty-sensitive rather than a robust 95% guarantee."
    elif report["target_above_any_interval"] and (req_x or req_y):
        color, bg, border = "#991b1b", "#fef2f2", "#ef4444"
        heading = "TARGET OUTSIDE THE 95% PROPAGATED INTERVAL"
        statement = "At least one selected target is above the corresponding 97.5th percentile and is not supported by the uncertainty propagation."
    elif req_x or req_y:
        color, bg, border = "#92400e", "#fffbeb", "#f59e0b"
        heading = "TARGET MET CONDITIONALLY; 95% ROBUSTNESS NOT ESTABLISHED"
        statement = "The conditional performance interval meets the selected target, but operating-window probability, numerical success, or joint target probability is below 95%."
    elif not (req_x or req_y) and report["nominal_inside_intervals"]:
        color, bg, border = "#166534", "#ecfdf5", "#22c55e"
        heading = "FINAL RESULT IS WITHIN THE 95% PROPAGATED INTERVAL"
        statement = "The nominal outlet abundances and recoveries all lie within their propagated 95% intervals."
    else:
        color, bg, border = "#1d4ed8", "#eff6ff", "#60a5fa"
        heading = "95% PROPAGATED PERFORMANCE INTERVAL"
        statement = "The intervals below quantify the entered independent thermodynamic and phase-flow uncertainties."

    nominal_note = (
        "The final nominal prediction lies inside all reported 95% intervals."
        if report["nominal_inside_intervals"]
        else "The final nominal prediction does not lie inside every reported 95% interval; inspect the nonlinear or truncated uncertainty model."
    )
    target_text = ""
    if req_x or req_y:
        requested = []
        if req_x:
            requested.append(f"x target = {float(target_x):.2f}%")
        if req_y:
            requested.append(f"y target = {float(target_y):.2f}%")
        target_text = f" Selected constraints: {', '.join(requested)}."
    probability_text = (
        f"Joint target probability: <b>{100.0 * report['joint_target_probability']:.1f}%</b>; "
        if req_x or req_y else ""
    )

    return f"""
    <div style="margin-top:12px; padding:13px 15px; border-radius:9px; background:{bg}; border:2px solid {border}; color:{color}; text-align:center;">
        <div style="font-size:14px; font-weight:800; letter-spacing:0.2px;">{heading}</div>
        <div style="font-size:12.5px; line-height:1.65; margin-top:5px;">{statement} {nominal_note}{target_text}</div>
        <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-top:9px; color:#1e293b;">
            <span style="background:#fff; border:1px solid #cbd5e1; border-radius:7px; padding:6px 9px;">x abundance: <b>{x_ci[0]:.2f}-{x_ci[1]:.2f}%</b></span>
            <span style="background:#fff; border:1px solid #cbd5e1; border-radius:7px; padding:6px 9px;">y abundance: <b>{y_ci[0]:.2f}-{y_ci[1]:.2f}%</b></span>
            <span style="background:#fff; border:1px solid #cbd5e1; border-radius:7px; padding:6px 9px;">x recovery: <b>{xr_ci[0]:.2f}-{xr_ci[1]:.2f}%</b></span>
            <span style="background:#fff; border:1px solid #cbd5e1; border-radius:7px; padding:6px 9px;">y recovery: <b>{yr_ci[0]:.2f}-{yr_ci[1]:.2f}%</b></span>
        </div>
        <div style="font-size:11.8px; color:#475569; margin-top:8px;">{probability_text}operating-window probability: <b>{100.0 * report['operability_probability']:.1f}%</b>; successful solves: <b>{report['successful_samples']}/{report['requested_samples']}</b>. Independent normal input models are truncated at their physical lower bounds; output intervals are conditional on successful solves.</div>
    </div>
    """

# ============================================================
# 6. Fast reverse designer: coarse screen -> exact refine -> local polish
# ============================================================
def make_fast_m_candidates(N):
    fracs = [0.12, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.88]
    vals = {1, N}
    vals.update(max(1, min(N, int(round(N * f)))) for f in fracs)
    return sorted(vals)

def score_design(x_pur, y_pur, x_rec, y_rec, F_vol, N):
    return (x_rec + y_rec) + 8.0 * F_vol - 0.02 * N

def candidate_passes(req_x, req_y, target_x, target_y, x_pur, y_pur):
    pass_x = (x_pur >= target_x) if req_x else True
    pass_y = (y_pur >= target_y) if req_y else True
    return pass_x and pass_y

def run_optimizer(
    req_x, req_y, target_x, target_y, fixed_N, design_F_vol, search_mode,
    F_conc, alpha_feed, beta, D_Sc, D_Ex, C_max,
    beta_err=0.0, D_Sc_err=0.0, D_Ex_err=0.0,
    W_err=0.0, F_err=0.0,
    progress=gr.Progress(),
):
    empty_file = gr.update()
    try:
        if not req_x and not req_y:
            return "<div class='panel-card' style='color:#b91c1c;'>Please select at least one purity target.</div>", empty_file

        target_x = float(target_x)
        target_y = float(target_y)
        fixed_N = int(fixed_N)
        design_F_vol = float(design_F_vol)
        F_conc = float(F_conc)
        alpha_feed = float(alpha_feed)
        beta = float(beta)
        D_Sc = float(D_Sc)
        D_Ex = float(D_Ex)
        C_max = float(C_max)
        beta_err, D_Sc_err, D_Ex_err, W_err, F_err, _ = validate_measurement_uncertainties(
            beta_err, D_Sc_err, D_Ex_err, W_err, F_err, 0.0
        )
        uncertainty_active = has_nonzero_uncertainty(
            beta_err, D_Sc_err, D_Ex_err, W_err, F_err
        )
        validate_common_inputs(
            N=fixed_N, F_vol=design_F_vol, F_conc=F_conc, alpha_feed=alpha_feed,
            beta=beta, D_Sc=D_Sc, D_Ex=D_Ex, C_max=C_max
        )
        if fixed_N < 10:
            return "<div class='panel-card' style='color:#b91c1c;'>The fixed total stage number must be at least 10.</div>", empty_file
        if design_F_vol <= 0:
            return "<div class='panel-card' style='color:#b91c1c;'>Preset feed flow rate must be greater than 0.</div>", empty_file

        N = fixed_N
        F_vol = design_F_vol
        mode = str(search_mode or "Fast")
        if mode == "Thorough":
            W_to_F_ratios = np.array([0.40, 0.50, 0.60, 0.75, 0.90, 1.05, 1.20, 1.40, 1.65, 1.90, 2.20, 2.60, 3.10, 3.70, 4.40, 5.20], dtype=float)
            s_fracs = np.array([0.05, 0.10, 0.16, 0.23, 0.31, 0.40, 0.50, 0.60, 0.69, 0.77, 0.84, 0.90, 0.95], dtype=float)
            refine_limit = 80
            m_offsets = [-6, -4, -2, 0, 2, 4, 6]
            polish_offsets = range(-8, 9)
            polish_frac_offsets = np.array([-0.10, -0.07, -0.04, -0.02, 0.0, 0.02, 0.04, 0.07, 0.10])
        else:
            W_to_F_ratios = np.array([0.45, 0.60, 0.75, 0.90, 1.10, 1.35, 1.65, 2.00, 2.50, 3.20, 4.00, 5.00], dtype=float)
            s_fracs = np.array([0.08, 0.16, 0.26, 0.38, 0.50, 0.62, 0.74, 0.84, 0.92], dtype=float)
            refine_limit = 45
            m_offsets = [-4, -2, 0, 2, 4]
            polish_offsets = range(-6, 7)
            polish_frac_offsets = np.array([-0.08, -0.04, -0.02, 0.0, 0.02, 0.04, 0.08])

        macro_points = []
        for wf in W_to_F_ratios:
            W_vol = F_vol * wf
            # Strict isotope-specific operating-window pre-screen.
            # D_Ex and D_Sc are measurable bulk distribution ratios; they are
            # converted to D_x,Ex and D_y,Sc using the feed isotope ratio.
            D_x_Ex_ref, _ = isotope_specific_distribution_from_total(D_Ex, beta, alpha_feed)
            _, D_y_Sc_ref = isotope_specific_distribution_from_total(D_Sc, beta, alpha_feed)
            F_crit = W_vol * (D_x_Ex_ref / D_y_Sc_ref - 1.0)
            if F_crit <= 0 or F_vol >= F_crit:
                continue
            S_min = (W_vol + F_vol) / D_x_Ex_ref
            S_max = W_vol / D_y_Sc_ref
            width = S_max - S_min
            if width <= 0:
                continue
            for frac in s_fracs:
                S_vol = S_min + frac * width
                operability_probability = 1.0
                if uncertainty_active:
                    screen = propagate_operability_uncertainty(
                        W_vol, F_vol, S_vol, alpha_feed, beta, D_Sc, D_Ex,
                        beta_err, D_Sc_err, D_Ex_err, W_err, F_err, 0.0,
                        sample_count=OPERABILITY_SCREEN_SAMPLE_COUNT,
                        random_seed=UNCERTAINTY_RANDOM_SEED + 17,
                    )
                    operability_probability = float(screen["feasibility_probability"])
                    if operability_probability < 0.95:
                        continue
                macro_points.append((
                    W_vol, S_vol, S_min, S_max, width, frac,
                    operability_probability,
                ))

        if not macro_points:
            return """
            <div class="panel-card" style="background:#fef2f2; border-color:#f87171;">
                <h3 style="color:#b91c1c; margin:0 0 10px 0;">No feasible operating window found</h3>
                <p style="font-size:13.5px; color:#1e293b;">At the fixed feed flow rate, no scanned scrubbing/organic-flow candidate retains at least 95% operating-window probability under the entered SDs.</p>
            </div>
            """, empty_file

        solve_cache = {}
        def eval_candidate(M, W_vol, S_vol, is_fine):
            key = (bool(is_fine), int(M), round(float(W_vol), 10), round(float(S_vol), 10))
            if key not in solve_cache:
                solve_cache[key] = ultra_fast_solve(
                    N, M, W_vol, F_vol, S_vol, F_conc, alpha_feed, beta, D_Sc, D_Ex, C_max,
                    target_x=target_x if req_x else 0, target_y=target_y if req_y else 0, is_fine=is_fine
                )
            return solve_cache[key]

        coarse_keep = []
        checked_coarse = 0
        mids = make_fast_m_candidates(N)
        total_coarse = max(len(mids) * len(macro_points), 1)
        for M in mids:
            for W_vol, S_vol, S_min, S_max, width, frac, operability_probability in macro_points:
                checked_coarse += 1
                if checked_coarse % 40 == 0:
                    progress(min(0.52 * checked_coarse / total_coarse, 0.52), desc="Fast coarse screening...")
                x_pur, y_pur, x_rec, y_rec = eval_candidate(M, W_vol, S_vol, is_fine=False)
                miss_x = max((target_x - x_pur) if req_x else 0.0, 0.0)
                miss_y = max((target_y - y_pur) if req_y else 0.0, 0.0)
                near_miss_penalty = 6.0 * (miss_x + miss_y)
                coarse_score = score_design(x_pur, y_pur, x_rec, y_rec, F_vol, N) - near_miss_penalty
                if coarse_score > -120:
                    coarse_keep.append((coarse_score, M, W_vol, S_vol, S_min, S_max, width, frac, operability_probability))

        if not coarse_keep:
            return """
            <div class="panel-card" style="background:#fef2f2; border-color:#f87171;">
                <h3 style="color:#b91c1c; margin:0 0 10px 0;">No promising configuration found</h3>
                <p style="font-size:13.5px; color:#1e293b;">The fast screen did not find any candidate close to the selected purity targets.</p>
            </div>
            """, empty_file

        coarse_keep.sort(key=lambda x: x[0], reverse=True)
        refine_seed_count = min(refine_limit, len(coarse_keep))
        refine_seeds = coarse_keep[:refine_seed_count]

        best_config = None
        best_score = -np.inf
        feasible_count = 0
        checked_fine = 0
        fine_jobs = refine_seed_count * 5
        seen = set()
        feasible_configs = []
        for _, M, W_vol, S_vol, S_min, S_max, width, frac, operability_probability in refine_seeds:
            local_Ms = sorted({max(1, min(N, M + d)) for d in m_offsets})
            for M2 in local_Ms:
                key = (M2, round(W_vol, 8), round(S_vol, 8))
                if key in seen:
                    continue
                seen.add(key)
                checked_fine += 1
                if checked_fine % 20 == 0:
                    progress(0.55 + min(0.30 * checked_fine / fine_jobs, 0.30), desc="Refining top candidates...")
                x_pur, y_pur, x_rec, y_rec = eval_candidate(M2, W_vol, S_vol, is_fine=True)
                if not candidate_passes(req_x, req_y, target_x, target_y, x_pur, y_pur):
                    continue
                feasible_count += 1
                fine_score = score_design(x_pur, y_pur, x_rec, y_rec, F_vol, N)
                if fine_score > best_score:
                    best_score = fine_score
                    best_config = {
                        "N": N, "M": M2, "W": W_vol, "F": F_vol, "S": S_vol,
                        "S_min": S_min, "S_max": S_max, "width": width,
                        "x_pur": x_pur, "y_pur": y_pur, "x_rec": x_rec, "y_rec": y_rec,
                        "score": fine_score, "operability_probability": operability_probability,
                    }
                feasible_configs.append({
                    "N": N, "M": M2, "W": W_vol, "F": F_vol, "S": S_vol,
                    "S_min": S_min, "S_max": S_max, "width": width,
                    "x_pur": x_pur, "y_pur": y_pur, "x_rec": x_rec, "y_rec": y_rec,
                    "score": fine_score, "operability_probability": operability_probability,
                })

        if best_config is not None:
            N = best_config["N"]
            F_vol = best_config["F"]
            W_vol = best_config["W"]
            S_min = best_config["S_min"]
            S_max = best_config["S_max"]
            width = best_config["width"]
            base_frac = (best_config["S"] - S_min) / max(width, 1e-30)
            local_fracs = np.clip(base_frac + polish_frac_offsets, 0.03, 0.97)
            local_Ms = sorted({max(1, min(N, best_config["M"] + d)) for d in polish_offsets})
            polish_jobs = len(local_fracs) * len(local_Ms)
            polish_done = 0
            for frac in local_fracs:
                S_vol = S_min + frac * width
                polish_operability_probability = best_config.get("operability_probability", 1.0)
                if uncertainty_active:
                    polish_screen = propagate_operability_uncertainty(
                        W_vol, F_vol, S_vol, alpha_feed, beta, D_Sc, D_Ex,
                        beta_err, D_Sc_err, D_Ex_err, W_err, F_err, 0.0,
                        sample_count=OPERABILITY_SCREEN_SAMPLE_COUNT,
                        random_seed=UNCERTAINTY_RANDOM_SEED + 17,
                    )
                    polish_operability_probability = float(polish_screen["feasibility_probability"])
                    if polish_operability_probability < 0.95:
                        continue
                for M in local_Ms:
                    polish_done += 1
                    if polish_done % 20 == 0:
                        progress(0.85 + min(0.14 * polish_done / polish_jobs, 0.14), desc="Local polishing...")
                    x_pur, y_pur, x_rec, y_rec = eval_candidate(M, W_vol, S_vol, is_fine=True)
                    if not candidate_passes(req_x, req_y, target_x, target_y, x_pur, y_pur):
                        continue
                    feasible_count += 1
                    fine_score = score_design(x_pur, y_pur, x_rec, y_rec, F_vol, N)
                    if fine_score > best_score:
                        best_score = fine_score
                        best_config.update({
                            "M": M, "S": S_vol, "x_pur": x_pur, "y_pur": y_pur,
                            "x_rec": x_rec, "y_rec": y_rec, "score": fine_score,
                            "operability_probability": polish_operability_probability,
                        })
                    feasible_configs.append({
                        "N": N, "M": M, "W": W_vol, "F": F_vol, "S": S_vol,
                        "S_min": S_min, "S_max": S_max, "width": width,
                        "x_pur": x_pur, "y_pur": y_pur, "x_rec": x_rec, "y_rec": y_rec,
                        "score": fine_score,
                        "operability_probability": polish_operability_probability,
                    })

        if best_config is None:
            return f"""
            <div class="panel-card" style="background:#fef2f2; border-color:#f87171;">
                <h3 style="color:#b91c1c; margin:0 0 10px 0;">No feasible configuration found</h3>
                <p style="font-size:13.5px; color:#1e293b;">The fixed-N, fixed-feed search screened <b>{checked_coarse}</b> coarse candidates and refined <b>{checked_fine}</b> top candidates, but none reached the selected purity constraints. Try relaxing the targets, increasing the fixed stage number, or expanding the flow grid.</p>
            </div>
            """, empty_file

        # Uncertainty is part of the reverse-design decision, not only a post-hoc label.
        # Validate the strongest distinct nominal candidates with common-random-number
        # Monte Carlo trials, prefer a robust 95% solution, and otherwise select the
        # design with the highest joint target probability.
        unique_configs = []
        seen_configs = set()
        for config in sorted(feasible_configs, key=lambda item: item["score"], reverse=True):
            config_key = (
                int(config["M"]), round(float(config["W"]), 9),
                round(float(config["S"]), 9),
            )
            if config_key in seen_configs:
                continue
            seen_configs.add(config_key)
            unique_configs.append(config)

        validation_count = min(4 if uncertainty_active else 1, len(unique_configs))
        performance_sample_count = (
            DESIGN_UNCERTAINTY_SAMPLE_COUNT_THOROUGH
            if mode == "Thorough"
            else DESIGN_UNCERTAINTY_SAMPLE_COUNT_FAST
        )
        validated_configs = []
        for index, config in enumerate(unique_configs[:validation_count]):
            progress(
                0.95 + 0.045 * (index + 1) / max(validation_count, 1),
                desc="Propagating uncertainty through finalist designs...",
            )
            report = propagate_design_performance_uncertainty(
                config["N"], config["M"], config["W"], config["F"], config["S"],
                F_conc, alpha_feed, beta, D_Sc, D_Ex, C_max,
                beta_err, D_Sc_err, D_Ex_err, W_err, F_err, 0.0,
                req_x=req_x, req_y=req_y, target_x=target_x, target_y=target_y,
                sample_count=performance_sample_count,
                random_seed=UNCERTAINTY_RANDOM_SEED + 101,
            )
            config = dict(config)
            config["uncertainty_report"] = report
            validated_configs.append(config)

        robust_configs = [
            config for config in validated_configs
            if config["uncertainty_report"]["robust_target_met"]
        ]
        if robust_configs:
            best_config = max(robust_configs, key=lambda item: item["score"])
        else:
            best_config = max(
                validated_configs,
                key=lambda item: (
                    item["uncertainty_report"]["joint_target_probability"],
                    item["uncertainty_report"]["solve_probability"],
                    item["score"],
                ),
            )

        uncertainty_report = best_config["uncertainty_report"]
        uncertainty_html = make_uncertainty_status_html(
            uncertainty_report, req_x, req_y, target_x, target_y
        )
        progress(1.0, desc="Done")

        s_pos = np.clip((best_config["S"] - best_config["S_min"]) / max(best_config["S_max"] - best_config["S_min"], 1e-30) * 100, 0, 100)
        window_bar = f"""
            <div style="margin:16px auto 14px auto; max-width:900px;">
                <div style="display:flex; justify-content:space-between; color:#64748b; font-size:12px; margin-bottom:5px;">
                    <span>S_min = {best_config['S_min']:.4f}</span>
                    <span>S_vol = {best_config['S']:.4f}</span>
                    <span>S_max = {best_config['S_max']:.4f}</span>
                </div>
                <div style="height:18px; position:relative; border-radius:9px; background:linear-gradient(90deg,#dcfce7,#bbf7d0); border:1px solid #86efac;">
                    <div style="position:absolute; left:{s_pos:.2f}%; top:-6px; width:4px; height:30px; background:#0f172a; border-radius:2px; transform:translateX(-50%);"></div>
                </div>
                <div style="font-size:12px; color:#64748b; margin-top:6px;">Recommended organic flow position within the effective operating window: <b>{s_pos:.1f}%</b> from lower to upper bound.</div>
            </div>
        """

        input_df = pd.DataFrame({
            "Parameter": [
                "Require target x abundance",
                "Target x abundance (%)",
                "Require target y abundance",
                "Target y abundance (%)",
                "Fixed total stages, N",
                "Preset feed flow rate, F_vol",
                "Total feed concentration, C_F",
                "Feed isotopic ratio, alpha_F",
                "Separation factor, beta",
                "Input standard deviation (SD) of beta",
                "Distribution ratio in scrubbing section, D_Sc",
                "Input standard deviation (SD) of D_Sc",
                "Distribution ratio in extraction section, D_Ex",
                "Input standard deviation (SD) of D_Ex",
                "Input standard deviation (SD) of W_vol",
                "Input standard deviation (SD) of F_vol",
                "Capacity limit, C_max",
                "Search mode",
            ],
            "Value": [
                bool(req_x),
                target_x,
                bool(req_y),
                target_y,
                fixed_N,
                design_F_vol,
                F_conc,
                alpha_feed,
                beta,
                beta_err,
                D_Sc,
                D_Sc_err,
                D_Ex,
                D_Ex_err,
                W_err,
                F_err,
                C_max,
                mode,
            ],
        })
        design_df = pd.DataFrame({
            "Parameter": [
                "Total stages, N",
                "Feed stage, M",
                "Scrubbing phase flow rate, W_vol",
                "Feed flow rate, F_vol",
                "Organic phase flow rate, S_vol",
                "S_min",
                "S_max",
                "Operating-window position from lower bound (%)",
                "Monte Carlo operating-window probability (%)",
                "Objective score",
            ],
            "Value": [
                best_config["N"],
                best_config["M"],
                best_config["W"],
                best_config["F"],
                best_config["S"],
                best_config["S_min"],
                best_config["S_max"],
                s_pos,
                100.0 * uncertainty_report["operability_probability"],
                best_config["score"],
            ],
        })
        performance_df = pd.DataFrame({
            "Metric": [
                "Predicted x abundance at organic outlet (%)",
                "Expected x recovery (%)",
                "Predicted y abundance at aqueous outlet (%)",
                "Expected y recovery (%)",
                "95% interval lower, x abundance (%)",
                "95% interval upper, x abundance (%)",
                "95% interval lower, y abundance (%)",
                "95% interval upper, y abundance (%)",
                "95% interval lower, x recovery (%)",
                "95% interval upper, x recovery (%)",
                "95% interval lower, y recovery (%)",
                "95% interval upper, y recovery (%)",
                "Joint target probability (%)",
                "Successful uncertainty solves (%)",
                "Robust 95% target satisfaction",
                "Nominal result inside all 95% intervals",
                "Uncertainty samples requested",
                "Coarse candidates screened",
                "Top candidates refined",
                "Feasible refined candidates",
            ],
            "Value": [
                best_config["x_pur"],
                best_config["x_rec"],
                best_config["y_pur"],
                best_config["y_rec"],
                uncertainty_report["intervals"]["x_pur"][0],
                uncertainty_report["intervals"]["x_pur"][1],
                uncertainty_report["intervals"]["y_pur"][0],
                uncertainty_report["intervals"]["y_pur"][1],
                uncertainty_report["intervals"]["x_rec"][0],
                uncertainty_report["intervals"]["x_rec"][1],
                uncertainty_report["intervals"]["y_rec"][0],
                uncertainty_report["intervals"]["y_rec"][1],
                100.0 * uncertainty_report["joint_target_probability"],
                100.0 * uncertainty_report["solve_probability"],
                uncertainty_report["robust_target_met"],
                uncertainty_report["nominal_inside_intervals"],
                uncertainty_report["requested_samples"],
                checked_coarse,
                checked_fine,
                feasible_count,
            ],
        })
        reverse_excel_path = os.path.join(tempfile.gettempdir(), f"reverse_design_N{best_config['N']}_M{best_config['M']}.xlsx")
        units_df = pd.DataFrame({
            "Quantity group": [
                "Volumetric flow rates",
                "Concentrations",
                "Dimensionless quantities",
                "Reverse-design uncertainty scope",
            ],
            "Requirement": [
                "W_vol, F_vol, S_vol, S_min, and S_max must use one common unit.",
                "C_F, C_max, and all calculated concentrations must use one common unit.",
                "alpha_F, beta, D_Sc, D_Ex, abundances, recoveries, and transport factors are dimensionless.",
                "S_vol is optimized deterministically; no SD is assigned to or propagated for S_vol.",
            ]
        })
        with pd.ExcelWriter(reverse_excel_path, engine="openpyxl") as writer:
            units_df.to_excel(writer, sheet_name="Unit convention", index=False)
            input_df.to_excel(writer, sheet_name="Input constraints", index=False)
            design_df.to_excel(writer, sheet_name="Optimized design", index=False)
            performance_df.to_excel(writer, sheet_name="Performance", index=False)

        summary_html = f"""
        <div class="panel-card" style="background:#f0fdf4; border-color:#86efac; border-top: 5px solid #16a34a;">
            <h3 style="color:#15803d; font-size:18px; margin:0 0 15px 0;">Reverse Design Completed</h3>
            <p style="font-size:12.8px; color:#334155; margin-top:-6px;">Fixed-N, fixed-feed <b>{mode}</b> heuristic search: screened <b>{checked_coarse}</b> coarse candidates, refined <b>{checked_fine}</b> top candidates, feasible refined candidates: <b>{feasible_count}</b>, uncertainty-validated finalists: <b>{validation_count}</b>.</p>
            <table style="width:100%; border-collapse: collapse; font-size:13.5px; text-align:center; margin-bottom:15px;">
                <tr style="background:#dcfce7; color:#166534; font-weight:bold;">
                    <td style="padding:10px; border:1px solid #bbf7d0;">Total Stages (N)</td>
                    <td style="padding:10px; border:1px solid #bbf7d0;">Feed Stage (M)</td>
                </tr>
                <tr>
                    <td style="padding:10px; border:1px solid #bbf7d0; font-size:22px; font-weight:700; color:#1e293b;">{best_config['N']}</td>
                    <td style="padding:10px; border:1px solid #bbf7d0; font-size:22px; font-weight:700; color:#1e293b;">{best_config['M']}</td>
                </tr>
            </table>
            <table style="width:100%; border-collapse: collapse; font-size:13.5px; text-align:center;">
                <tr style="background:#e0e7ff; color:#1e40af; font-weight:bold;">
                    <td style="padding:10px; border:1px solid #bfdbfe;">Scrubbing, {var("W", "vol")}</td>
                    <td style="padding:10px; border:1px solid #bfdbfe;">Feed, {var("F", "vol")}</td>
                    <td style="padding:10px; border:1px solid #bfdbfe;">Organic, {var("S", "vol")}</td>
                </tr>
                <tr>
                    <td style="padding:10px; border:1px solid #bfdbfe; font-size:20px; font-weight:700; color:#1e293b;">{best_config['W']:.4f}</td>
                    <td style="padding:10px; border:1px solid #bfdbfe; font-size:20px; font-weight:700; color:#1e293b;">{best_config['F']:.4f}</td>
                    <td style="padding:10px; border:1px solid #bfdbfe; font-size:20px; font-weight:700; color:#1e293b;">{best_config['S']:.4f}</td>
                </tr>
            </table>
            <div style="font-size:12.8px; color:#334155; margin:10px auto 14px auto;">
                Operating window: S_min = <b>{best_config['S_min']:.4f}</b>, S_max = <b>{best_config['S_max']:.4f}</b>; objective score = <b>{best_config['score']:.3f}</b>.
            </div>
            {window_bar}
            <div style="display:flex; gap:10px; justify-content:center; margin-top:15px;">
                <div style="flex:1; background:#ffffff; border:1px solid #fecaca; border-radius:8px; padding:12px;">
                    <b style="color:#b91c1c;">Target x (Organic Outlet)</b><br>
                    Predicted Abundance: <b>{best_config['x_pur']:.2f}%</b><br>
                    Expected Recovery: <b>{best_config['x_rec']:.2f}%</b>
                </div>
                <div style="flex:1; background:#ffffff; border:1px solid #bfdbfe; border-radius:8px; padding:12px;">
                    <b style="color:#1d4ed8;">Target y (Aqueous Outlet)</b><br>
                    Predicted Abundance: <b>{best_config['y_pur']:.2f}%</b><br>
                    Expected Recovery: <b>{best_config['y_rec']:.2f}%</b>
                </div>
            </div>
            {uncertainty_html}
        </div>
        """
        return summary_html, reverse_excel_path
    except Exception as e:
        return f"<div class='panel-card' style='color:#b91c1c;'>Optimization Error: {str(e)}</div>", empty_file

# ============================================================
# 7. Operability-window display and map
# ============================================================
def make_operability_map(
    W, F, S, alpha_ref, beta, D_Sc, D_Ex,
    beta_err=0.0, D_Sc_err=0.0, D_Ex_err=0.0,
    W_err=0.0, F_err=0.0, S_err=0.0,
):
    fig, ax = plt.subplots(figsize=(8.2, 3.75))
    try:
        if any(val is None for val in [W, F, S, beta, D_Sc, D_Ex]):
            raise ValueError("Incomplete inputs")

        W, F, S, alpha_ref, beta, D_Sc, D_Ex = map(float, [W, F, S, alpha_ref, beta, D_Sc, D_Ex])
        beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err = validate_measurement_uncertainties(
            beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err
        )
        has_uncertainty = has_nonzero_uncertainty(
            beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err
        )
        uncertainty = None
        if has_uncertainty:
            uncertainty = propagate_operability_uncertainty(
                W, F, S, alpha_ref, beta, D_Sc, D_Ex,
                beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err,
                return_samples=True,
            )

        S_min, S_max, width, _, _, _ = compute_window(W, F, S, alpha_ref, beta, D_Sc, D_Ex)

        D_x_Ex_ref, _ = isotope_specific_distribution_from_total(D_Ex, beta, alpha_ref)
        _, D_y_Sc_ref = isotope_specific_distribution_from_total(D_Sc, beta, alpha_ref)

        # Strict critical feed flow mapping: window closes when S_min >= S_max.
        F_crit = W * (D_x_Ex_ref / D_y_Sc_ref - 1.0)

        if F_crit > 0: F_upper = max(2.5 * F, 1.15 * F_crit, 0.1)
        else: F_upper = max(2.5 * F, W, 0.1)

        S_upper = max(1.6 * S, 1.35 * S_max, 0.1)
        if uncertainty is not None:
            F_upper = max(F_upper, 1.15 * max(float(uncertainty["F_crit"][1]), 0.0))
            S_upper = max(
                S_upper,
                1.20 * float(uncertainty["S_max"][1]),
                1.20 * float(uncertainty["S"][1]),
            )
        F_grid = np.linspace(0, F_upper, 260)
        S_grid = np.linspace(0, S_upper, 260)
        F_mesh, S_mesh = np.meshgrid(F_grid, S_grid)
        
        # Strict isotope-specific boundary logic.
        S_min_grid = (W + F_mesh) / D_x_Ex_ref
        S_max_grid = W / D_y_Sc_ref

        region = np.where(F_mesh >= F_crit, 3, np.where(S_mesh < S_min_grid, 0, np.where(S_mesh > S_max_grid, 2, 1)))

        cmap = ListedColormap(["#fee2e2", "#dcfce7", "#ede9fe", "#e2e8f0"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        ax.contourf(F_mesh, S_mesh, region, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=cmap, norm=norm, alpha=0.92)

        if F_crit > 0:
            F_line_grid = np.linspace(0, min(F_crit, F_upper), 220)
            S_min_line = (W + F_line_grid) / D_x_Ex_ref

            if uncertainty is not None:
                raw = uncertainty["samples"]
                if F_err > 0:
                    standardized_F_error = (raw["F"] - F) / F_err
                else:
                    standardized_F_error = np.zeros_like(raw["F"])
                realized_F_grid = np.maximum(
                    F_line_grid[None, :] + F_err * standardized_F_error[:, None],
                    0.0,
                )
                S_min_curves = (
                    raw["W"][:, None] + realized_F_grid
                ) / uncertainty["D_x_Ex_samples"][:, None]
                S_min_low, S_min_high = np.quantile(
                    S_min_curves, UNCERTAINTY_INTERVAL, axis=0
                )
                S_max_low, S_max_high = map(float, uncertainty["S_max"])

                ax.fill_between(
                    F_line_grid, S_min_low, S_min_high,
                    color="#475569", alpha=0.20, linewidth=0, zorder=3,
                )
                ax.fill_between(
                    F_line_grid,
                    np.full_like(F_line_grid, S_max_low),
                    np.full_like(F_line_grid, S_max_high),
                    color="#475569", alpha=0.20, linewidth=0, zorder=3,
                )

            ax.plot(F_line_grid, S_min_line, color="#b91c1c", linewidth=2.2, zorder=4)
            ax.plot([0, min(F_crit, F_upper)], [S_max, S_max], color="#6d28d9", linewidth=2.2, zorder=4)

        if uncertainty is not None:
            F_crit_low, F_crit_high = map(float, uncertainty["F_crit"])
            span_low = max(0.0, F_crit_low)
            span_high = min(F_upper, F_crit_high)
            if span_high > span_low:
                ax.axvspan(
                    span_low, span_high,
                    color="#64748b", alpha=0.13, linewidth=0, zorder=2.8,
                )

        if 0 < F_crit < F_upper:
            ax.axvline(F_crit, color="#475569", linestyle="--", linewidth=1.5, zorder=4)

        if uncertainty is not None and (F_err > 0 or S_err > 0):
            xerr = np.array([[max(F - float(uncertainty["F"][0]), 0.0)], [max(float(uncertainty["F"][1]) - F, 0.0)]])
            yerr = np.array([[max(S - float(uncertainty["S"][0]), 0.0)], [max(float(uncertainty["S"][1]) - S, 0.0)]])
            ax.errorbar(
                [F], [S], xerr=xerr, yerr=yerr, fmt="none",
                ecolor="#0f172a", elinewidth=1.2, capsize=3, alpha=0.85, zorder=4.8,
            )
        ax.scatter([F], [S], color="#0f172a", edgecolor="white", linewidth=0.8, s=58, zorder=5, label="Current point")
        ax.annotate("Current point", xy=(F, S), xytext=(8, 8), textcoords="offset points", fontsize=8.5, ha="left", color="#0f172a")

        ax.set_xlim(0, F_upper)
        ax.set_ylim(0, S_upper)
        ax.set_xlabel(r"$F_{\mathrm{vol}}$ (consistent flow units)")
        ax.set_ylabel(r"$S_{\mathrm{vol}}$ (consistent flow units)")
        ax.set_title(r"$F_{\mathrm{vol}}$–$S_{\mathrm{vol}}$ operability map", pad=8)

        legend_elements = [
            Line2D([0], [0], color="#6d28d9", lw=2.2, label=r"$S_{\max}$"),
            Line2D([0], [0], color="#b91c1c", lw=2.2, label=r"$S_{\min}$"),
            Patch(facecolor="#fee2e2", edgecolor="none", label="Total stripping"),
            Patch(facecolor="#dcfce7", edgecolor="none", label="Effective Separation"),
            Patch(facecolor="#ede9fe", edgecolor="none", label="Total extraction"),
            Patch(facecolor="#e2e8f0", edgecolor="none", label="Infeasible domain"),
        ]
        
        if uncertainty is not None:
            legend_elements.append(
                Patch(
                    facecolor="#475569", edgecolor="none", alpha=0.20,
                    label="95% propagated intervals",
                )
            )

        if 0 < F_crit < F_upper:
             legend_elements.append(Line2D([0], [0], color="#475569", linestyle="--", lw=1.5, label=r"Critical feed ($F_{\max}$)"))

        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=7.3,
            frameon=True,
            framealpha=0.92,
            ncol=2 if uncertainty is not None else 1,
            columnspacing=1.0,
        )
        ax.grid(True, alpha=0.22)
        for spine in ax.spines.values(): spine.set_color("#94a3b8")
        fig.tight_layout()

    except Exception as e:
        ax.text(0.5, 0.5, f"Waiting for valid inputs...\n{str(e)}", ha="center", va="center", fontsize=10.5, color="#64748b")
        ax.set_axis_off()
    return fig

def update_window_outputs(
    W, F, S, alpha_ref, beta, D_Sc, D_Ex,
    beta_err=0.0, D_Sc_err=0.0, D_Ex_err=0.0,
    W_err=0.0, F_err=0.0, S_err=0.0,
):
    try:
        if any(x is None for x in [W, F, S, alpha_ref, beta, D_Sc, D_Ex]):
            raise ValueError("Please provide all numerical parameters.")

        beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err = validate_measurement_uncertainties(
            beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err
        )
        S_min, S_max, width, E_x_ext, E_y_scrub, regime = compute_window(W, F, S, alpha_ref, beta, D_Sc, D_Ex)
        W, F, S, alpha_ref, beta, D_Sc, D_Ex = map(float, [W, F, S, alpha_ref, beta, D_Sc, D_Ex])
        D_x_Ex_ref, _ = isotope_specific_distribution_from_total(D_Ex, beta, alpha_ref)
        _, D_y_Sc_ref = isotope_specific_distribution_from_total(D_Sc, beta, alpha_ref)

        if width <= 0:
            status_color, status_bg = "#b91c1c", "#fef2f2"
            status_text = "No feasible window is available. Under the current distribution ratios and flow rates, the lower bound is not smaller than the upper bound."
            bar_html, lower_margin, upper_margin = "", np.nan, np.nan
        else:
            if S < S_min:
                status_color, status_bg = "#b91c1c", "#fef2f2"
                status_text = "Infeasible: the organic-phase flow rate is below the lower bound, so x cannot be sufficiently transported into the organic phase in the extraction section."
            elif S > S_max:
                status_color, status_bg = "#6d28d9", "#f5f3ff"
                status_text = "Infeasible: the organic-phase flow rate exceeds the upper bound, so y is excessively extracted in the scrubbing section."
            else:
                status_color, status_bg = "#15803d", "#ecfdf5"
                status_text = "Feasible: the current organic-phase flow rate lies inside the effective operating window."

            lower_margin = (S - S_min) / S_min * 100 if S_min > 0 else np.nan
            upper_margin = (S_max - S) / S_max * 100 if S_max > 0 else np.nan

            display_min = min(S_min - 0.8 * width, S)
            display_max = max(S_max + 0.8 * width, S)
            display_range = max(display_max - display_min, 1e-12)

            left_pct = np.clip(((S_min - display_min) / display_range) * 100, 0, 100)
            right_pct = np.clip(((S_max - display_min) / display_range) * 100, 0, 100)
            s_pct = np.clip(((S - display_min) / display_range) * 100, 0, 100)

            bar_html = f"""
            <div style="margin:40px auto 72px auto; position:relative; height:50px; background:#e2e8f0; border-radius:9px; overflow:visible; width:96%; border:1px solid #dbe3ef;">
                <div style="position:absolute; left:0; top:0; width:{left_pct:.2f}%; height:100%; background:#fecaca; border-radius:9px 0 0 9px; display:flex; align-items:center; justify-content:center;">
                    <span style="font-size:11.5px; color:#991b1b; font-weight:740;">Total stripping</span>
                </div>
                <div style="position:absolute; left:{left_pct:.2f}%; top:0; width:{max(right_pct - left_pct, 0):.2f}%; height:100%; background:#bbf7d0; border-left:2px dashed #16a34a; border-right:2px dashed #16a34a; display:flex; align-items:center; justify-content:center;">
                    <span style="font-size:11.5px; color:#166534; font-weight:740;">Effective operating</span>
                </div>
                <div style="position:absolute; left:{right_pct:.2f}%; top:0; width:{max(100 - right_pct, 0):.2f}%; height:100%; background:#ddd6fe; border-radius:0 9px 9px 0; display:flex; align-items:center; justify-content:center;">
                    <span style="font-size:11.5px; color:#5b21b6; font-weight:740;">Total extraction</span>
                </div>
                <div style="position:absolute; left:{s_pct:.2f}%; top:-16px; width:4px; height:82px; background:#0f172a; transform:translateX(-50%); box-shadow:0 0 5px rgba(15,23,42,0.35);"></div>
                <div style="position:absolute; left:{s_pct:.2f}%; top:-42px; transform:translateX(-50%); background:#ffffff; border:2px solid #0f172a; padding:3px 10px; border-radius:14px; font-weight:740; font-size:12.5px; color:#0f172a; white-space:nowrap;">Current {var("S", "vol")} = {S:.4f}</div>
                <div style="position:absolute; left:{left_pct:.2f}%; top:60px; transform:translateX(-50%); font-size:12.5px; color:#334155; font-weight:740; white-space:nowrap;">{var("S", "min")} = {S_min:.4f}</div>
                <div style="position:absolute; left:{right_pct:.2f}%; top:60px; transform:translateX(-50%); font-size:12.5px; color:#334155; font-weight:740; white-space:nowrap;">{var("S", "max")} = {S_max:.4f}</div>
            </div>
            """

        margin_html = ""
        if width > 0:
            margin_html = f"""
            <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-top:8px;">
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; color:#334155; min-width:138px;">Lower-bound margin<br><b>{lower_margin:.2f}%</b></div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; color:#334155; min-width:138px;">Upper-bound margin<br><b>{upper_margin:.2f}%</b></div>
            </div>
            """

        uncertainty_html = ""
        if has_nonzero_uncertainty(
            beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err
        ):
            uncertainty = propagate_operability_uncertainty(
                W, F, S, alpha_ref, beta, D_Sc, D_Ex,
                beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err,
            )
            probability = float(uncertainty["feasibility_probability"])
            if probability >= 0.95:
                u_color, u_bg, u_border = "#166534", "#ecfdf5", "#22c55e"
                u_heading = "95% ROBUST OPERATING POINT"
                u_statement = "At least 95% of the propagated thermodynamic and phase-flow samples remain inside the effective operating window."
            elif S_min < S < S_max:
                u_color, u_bg, u_border = "#92400e", "#fffbeb", "#f59e0b"
                u_heading = "NOMINALLY FEASIBLE, BUT UNCERTAINTY-SENSITIVE"
                u_statement = "The nominal point is feasible, but fewer than 95% of propagated samples remain inside the operating window."
            else:
                u_color, u_bg, u_border = "#991b1b", "#fef2f2", "#ef4444"
                u_heading = "NOT ROBUST AT THE 95% PROBABILITY THRESHOLD"
                u_statement = "The entered uncertainty model does not support a robust effective-window classification."
            uncertainty_html = f"""
            <div style="margin-top:12px; padding:12px 14px; border-radius:9px; background:{u_bg}; border:2px solid {u_border}; color:{u_color};">
                <div style="font-size:14px; font-weight:800;">{u_heading}</div>
                <div style="font-size:12.5px; line-height:1.6; margin-top:4px;">{u_statement}</div>
                <div style="font-size:12px; color:#334155; margin-top:7px;">
                    Operating-window probability: <b>{100.0 * probability:.1f}%</b>;
                    95% intervals: {var("S", "min")} = <b>{uncertainty['S_min'][0]:.4f}-{uncertainty['S_min'][1]:.4f}</b>,
                    current {var("S", "vol")} = <b>{uncertainty['S'][0]:.4f}-{uncertainty['S'][1]:.4f}</b>,
                    {var("S", "max")} = <b>{uncertainty['S_max'][0]:.4f}-{uncertainty['S_max'][1]:.4f}</b>
                    ({uncertainty['sample_count']} trials).
                </div>
            </div>
            """

        html_content = f"""
        <div class="panel-card status-card">
            <h3 style="margin:0 0 8px 0; color:#0f172a; font-size:16px; font-weight:760;">Operating-window analyzer</h3>
            <div style="font-size:12.5px; color:#64748b; margin-bottom:10px; line-height:1.65;">
                Strict isotope-specific feasibility criterion: {var("S", "min")} &lt; {var("S", "vol")} &lt; {var("S", "max")},
                where
                {var("S", "min")} = ({var("W", "vol")} + {var("F", "vol")}) / <i>D</i><sub>x,Ex</sub>
                and
                {var("S", "max")} = {var("W", "vol")} / <i>D</i><sub>y,Sc</sub>.
                The bulk ratios are converted using {var("α", "F")}:
                <i>D</i><sub>x,Ex</sub> = {var("β")}(1+{var("α", "F")})<i>D</i><sub>Ex</sub>/(1+{var("β")}{var("α", "F")});
                <i>D</i><sub>y,Sc</sub> = (1+{var("α", "F")})<i>D</i><sub>Sc</sub>/(1+{var("β")}{var("α", "F")}).
            </div>
            <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-bottom:11px;">
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; min-width:105px;">{var("S", "min")}<br><b>{S_min:.4f}</b></div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; min-width:105px;">{var("S", "max")}<br><b>{S_max:.4f}</b></div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; min-width:118px;">Window width<br><b>{width:.4f}</b></div>
                <div style="background:{status_bg}; border:1px solid {status_color}; border-radius:8px; padding:9px 15px; min-width:190px;">Predicted regime<br><b style="color:{status_color};">{regime}</b></div>
            </div>
            <div style="background:{status_bg}; color:{status_color}; border:1px solid {status_color}; border-radius:8px; padding:8px 12px; font-weight:720; margin:10px auto; max-width:1000px; font-size:12.8px;">{status_text}</div>
            {bar_html}
            <div style="display:flex; justify-content:center; gap:16px; flex-wrap:wrap; margin-top:10px;">
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:11px 16px; min-width:285px;">Extraction-section transport factor for isotope x<br>{var("E", "x,ext")} &gt; 1 required<br><b style="font-size:19px; color:{'#15803d' if E_x_ext > 1 else '#b91c1c'};">{E_x_ext:.4f}</b></div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:11px 16px; min-width:285px;">Scrubbing-section transport factor for isotope y<br>{var("E", "y,scrub")} &lt; 1 required<br><b style="font-size:19px; color:{'#15803d' if E_y_scrub < 1 else '#b91c1c'};">{E_y_scrub:.4f}</b></div>
            </div>
            {margin_html}
            {uncertainty_html}
        </div>
        """
        fig_map = make_operability_map(
            W, F, S, alpha_ref, beta, D_Sc, D_Ex,
            beta_err, D_Sc_err, D_Ex_err,
            W_err, F_err, S_err,
        )
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
        fig = make_operability_map(1.0, 0.1, 0.8, 1.0, 1.02, 0.5, 1.6)
        return error_html, fig

# ============================================================
# 8. Main TDMA Solver 
# ============================================================
def solve_fractional_extraction(
    N, M, W_vol, F_vol, S_vol, F_conc, alpha_feed, beta, D_Sc, D_Ex, C_max,
    beta_err=0.0, D_Sc_err=0.0, D_Ex_err=0.0,
    W_err=0.0, F_err=0.0, S_err=0.0,
):
    try:
        validate_common_inputs(
            N=N, M=M, W_vol=W_vol, F_vol=F_vol, S_vol=S_vol, F_conc=F_conc,
            alpha_feed=alpha_feed, beta=beta, D_Sc=D_Sc, D_Ex=D_Ex, C_max=C_max
        )
        N, M = int(N), int(M)
        W_vol, F_vol, S_vol = map(float, [W_vol, F_vol, S_vol])
        F_conc, alpha_feed, beta = map(float, [F_conc, alpha_feed, beta])
        D_Sc, D_Ex, C_max_val = map(float, [D_Sc, D_Ex, C_max])
        beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err = validate_measurement_uncertainties(
            beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err
        )

        feed_index = M - 1
        stages = np.arange(1, N + 1)
        W = np.zeros(N)
        S = np.full(N, S_vol)
        D_ideal = np.zeros(N)

        for i in range(N):
            if i < feed_index:
                W[i] = W_vol
                D_ideal[i] = D_Sc
            else:
                W[i] = W_vol + F_vol
                D_ideal[i] = D_Ex

        eps = 1e-30
        lower, diag, upper, rhs = build_tridiagonal_system(W, S, D_ideal, feed_index, F_vol * F_conc)
        A_total = thomas_solve(lower, diag, upper, rhs)

        sat_errors = []
        for _ in range(8000):
            D_eff = np.zeros(N)
            for i in range(N):
                O_theoretical = D_ideal[i] * A_total[i]
                if O_theoretical > C_max_val: D_eff[i] = C_max_val / (A_total[i] + eps)
                else: D_eff[i] = D_ideal[i]

            lower, diag, upper, rhs = build_tridiagonal_system(W, S, D_eff, feed_index, F_vol * F_conc)
            A_new = np.maximum(thomas_solve(lower, diag, upper, rhs), 0.0)

            err = np.max(np.abs(A_new - A_total) / (np.abs(A_total) + eps))
            sat_errors.append(err)
            
            if err > 1.0: damping_sat = 0.005
            elif err > 0.1: damping_sat = 0.02
            elif err > 0.01: damping_sat = 0.05
            else: damping_sat = 0.15

            A_total = damping_sat * A_new + (1.0 - damping_sat) * A_total
            if err < 1e-12: break

        D_eff = np.zeros(N)
        for i in range(N):
            if D_ideal[i] * A_total[i] > C_max_val: D_eff[i] = C_max_val / (A_total[i] + eps)
            else: D_eff[i] = D_ideal[i]
        O_total = D_eff * A_total

        z_x = F_conc * alpha_feed / (1.0 + alpha_feed)
        z_y = F_conc - z_x
        a = A_total * alpha_feed / (1.0 + alpha_feed)

        errors_history = []
        for _ in range(8000):
            D_x = (beta * O_total) / (A_total + a * (beta - 1.0) + eps)
            lower_x, diag_x, upper_x, rhs_x = build_tridiagonal_system(W, S, D_x, feed_index, F_vol * z_x)
            a_pred = np.clip(thomas_solve(lower_x, diag_x, upper_x, rhs_x), 0.0, A_total)

            rel_error = np.max(np.abs(a_pred - a) / (np.abs(a) + eps))
            errors_history.append(rel_error)

            if rel_error > 1.0: damping_iso = 0.005
            elif rel_error > 0.1: damping_iso = 0.02
            elif rel_error > 0.01: damping_iso = 0.05
            else: damping_iso = 0.15

            a = damping_iso * a_pred + (1.0 - damping_iso) * a
            a = np.clip(a, 0.0, A_total)
            if rel_error < 1e-12: break

        D_x = (beta * O_total) / (A_total + a * (beta - 1.0) + eps)
        D_y = D_x / beta

        b = np.clip(D_x * a, 0.0, O_total)
        c = np.maximum(A_total - a, 0.0)
        d = np.maximum(O_total - b, 0.0)

        x_abundance_org = b / (b + d + eps)
        y_abundance_aq = c / (a + c + eps)
        x_recovery_org = (S_vol * b) / (F_vol * z_x + eps)
        y_recovery_aq = ((W_vol + F_vol) * c) / (F_vol * z_y + eps)

        E_x = S * D_x / (W + eps)
        E_y = S * D_y / (W + eps)

        mb_x, mb_y = [], []
        for i in range(N):
            in_x = ((W[i - 1] * a[i - 1] if i > 0 else 0) + (S[i + 1] * b[i + 1] if i < N - 1 else 0) + (F_vol * z_x if i == feed_index else 0))
            in_y = ((W[i - 1] * c[i - 1] if i > 0 else 0) + (S[i + 1] * d[i + 1] if i < N - 1 else 0) + (F_vol * z_y if i == feed_index else 0))
            out_x = W[i] * a[i] + S[i] * b[i]
            out_y = W[i] * c[i] + S[i] * d[i]
            mb_x.append(abs(in_x - out_x) / (abs(in_x) + eps))
            mb_y.append(abs(in_y - out_y) / (abs(in_y) + eps))

        mb_x, mb_y = np.maximum(np.array(mb_x), 1e-16), np.maximum(np.array(mb_y), 1e-16)
        eq_x = np.maximum(np.abs(b - D_x * a) / (np.abs(b) + np.abs(D_x * a) + eps), 1e-16)
        eq_y = np.maximum(np.abs(d - D_y * c) / (np.abs(d) + np.abs(D_y * c) + eps), 1e-16)
        beta_residual = np.maximum(np.abs((D_x / (D_y + eps)) - beta) / (beta + eps), 1e-16)
        errors_history = np.maximum(np.array(errors_history), 1e-16)

        feed_x_flux = F_vol * z_x
        feed_y_flux = F_vol * z_y
        outlet_x_flux = S_vol * b[0] + (W_vol + F_vol) * a[-1]
        outlet_y_flux = S_vol * d[0] + (W_vol + F_vol) * c[-1]
        closure_x = abs(outlet_x_flux - feed_x_flux) / (abs(feed_x_flux) + eps)
        closure_y = abs(outlet_y_flux - feed_y_flux) / (abs(feed_y_flux) + eps)

        S_min, S_max, width, E_x_ext, E_y_scrub, regime = compute_window(W_vol, F_vol, S_vol, alpha_feed, beta, D_Sc, D_Ex)

        peak_O = np.max(O_total)
        is_saturated = (peak_O >= C_max_val * 0.999)
        cap_text = f"{C_max_val:.4g}"

        sat_color = "#c2410c" if is_saturated else "#15803d"
        sat_bg = "#fff7ed" if is_saturated else "#ecfdf5"
        sat_border = "#fed7aa" if is_saturated else "#bbf7d0"
        sat_icon = "⚠️ Capacity limit active" if is_saturated else "✅ Capacity limit inactive"

        saturation_html = f"""
        <div style="margin-top:12px; padding:8px 12px; border-radius:8px; background:{sat_bg}; border:1px solid {sat_border}; color:{sat_color}; font-size:12.8px; text-align:center;">
            <b style="font-size:13.5px;">{sat_icon}</b>: peak organic-phase concentration is <b>{peak_O:.4e}</b> concentration units
            (capacity limit: <b>{cap_text}</b> concentration units).
        </div>
        """

        closure_color = "#15803d" if max(closure_x, closure_y) < 1e-8 else "#c2410c"
        closure_bg = "#ecfdf5" if max(closure_x, closure_y) < 1e-8 else "#fff7ed"
        closure_html = f"""
        <div style="margin-top:10px; padding:8px 12px; border-radius:8px; background:{closure_bg}; border:1px solid #e2e8f0; color:{closure_color}; font-size:12.8px; text-align:center;">
            Global outlet mass closure: isotope x = <b>{closure_x:.2e}</b>, isotope y = <b>{closure_y:.2e}</b>.
        </div>
        """

        performance_uncertainty_report = None
        performance_uncertainty_html = ""
        if has_nonzero_uncertainty(
            beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err
        ):
            performance_uncertainty_report = propagate_design_performance_uncertainty(
                N, M, W_vol, F_vol, S_vol, F_conc, alpha_feed,
                beta, D_Sc, D_Ex, C_max_val,
                beta_err, D_Sc_err, D_Ex_err, W_err, F_err, S_err,
                sample_count=DESIGN_UNCERTAINTY_SAMPLE_COUNT_FAST,
                random_seed=UNCERTAINTY_RANDOM_SEED + 211,
            )
            performance_uncertainty_html = make_uncertainty_status_html(
                performance_uncertainty_report
            )

        summary_html = f"""
        <div class="panel-card">
            <h3 style="margin:0 0 10px 0; color:#0f172a; font-size:16px; font-weight:760;">Separation performance summary</h3>
            <div style="margin:0 auto 15px auto; color:#334155; line-height:1.7; font-size:12.8px;">
                Feed composition:
                {var("C", "F")} = <b>{F_conc:.4g}</b> concentration units;
                {var("C", "x,F")} = <b>{z_x:.4e}</b> concentration units;
                {var("C", "y,F")} = <b>{z_y:.4e}</b> concentration units;
                {var("α", "F")} = {var("C", "x,F")}/{var("C", "y,F")} = <b>{alpha_feed:.4g}</b>.
            </div>
            <div style="display:flex; justify-content:center; gap:18px; flex-wrap:wrap;">
                <div style="flex:1; min-width:320px; max-width:520px; background:#ffffff;
                            border:1px solid #fecaca; border-top:4px solid #dc2626;
                            border-radius:9px; padding:15px; text-align:center;">
                    <h4 style="margin:0 0 8px 0; color:#b91c1c; font-size:14px;">Loaded organic outlet, Stage 1</h4>
                    <div style="line-height:1.9; color:#1e293b; font-size:12.8px;">
                        Isotope x concentration:
                        <b>{b[0]:.4e}</b> concentration units<br>
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
                        <b>{c[-1]:.4e}</b> concentration units<br>
                        Isotope y abundance:
                        <b style="color:#1d4ed8;">{y_abundance_aq[-1] * 100:.4f}%</b><br>
                        Isotope y recovery:
                        <b style="color:#1d4ed8;">{y_recovery_aq[-1] * 100:.4f}%</b>
                    </div>
                </div>
            </div>
            <div style="margin-top:15px; color:#64748b; font-size:12.5px; line-height:1.7;">
                Predicted operating regime: <b>{regime}</b>. Calculated bounds: {var("S", "min")} = <b>{S_min:.4f}</b>, {var("S", "max")} = <b>{S_max:.4f}</b>.
            </div>
            {saturation_html}
            {closure_html}
            {performance_uncertainty_html}
        </div>
        """

        fig_prof, ax_prof = plt.subplots(3, 1, figsize=(7.2, 9.4))
        ax_prof = np.ravel(ax_prof)
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
        ax_prof[1].set_ylabel(r"Concentration (consistent units)")
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
        fig_prof.tight_layout(pad=2.0)

        fig_val, ax_val = plt.subplots(3, 1, figsize=(7.2, 9.4))
        ax_val = np.ravel(ax_val)
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
        fig_val.tight_layout(pad=2.0)

        stage_df = pd.DataFrame({
            "Stage": stages,
            "Aqueous x, a_i": a,
            "Organic x, b_i": b,
            "Aqueous y, c_i": c,
            "Organic y, d_i": d,
            "Total Aqueous, A_i": A_total,
            "Total Organic, O_i": O_total,
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
                "Input standard deviation (SD) of beta",
                "Distribution ratio in scrubbing section, D_Sc",
                "Input standard deviation (SD) of D_Sc",
                "Distribution ratio in extraction section, D_Ex",
                "Input standard deviation (SD) of D_Ex",
                "Scrubbing-phase flow rate, W_vol",
                "Input standard deviation (SD) of W_vol",
                "Feed flow rate, F_vol",
                "Input standard deviation (SD) of F_vol",
                "Organic-phase flow rate, S_vol",
                "Input standard deviation (SD) of S_vol",
            ],
            "Value": [
                N, M, F_conc, alpha_feed, C_max_val,
                beta, beta_err, D_Sc, D_Sc_err, D_Ex, D_Ex_err,
                W_vol, W_err, F_vol, F_err, S_vol, S_err
            ]
        })

        uncertainty_df = None
        if performance_uncertainty_report is not None:
            u = performance_uncertainty_report
            uncertainty_df = pd.DataFrame({
                "Metric": [
                    "x abundance at organic outlet (%)",
                    "y abundance at aqueous outlet (%)",
                    "x recovery at organic outlet (%)",
                    "y recovery at aqueous outlet (%)",
                ],
                "Nominal": [
                    u["nominal"]["x_pur"], u["nominal"]["y_pur"],
                    u["nominal"]["x_rec"], u["nominal"]["y_rec"],
                ],
                "95% interval lower": [
                    u["intervals"]["x_pur"][0], u["intervals"]["y_pur"][0],
                    u["intervals"]["x_rec"][0], u["intervals"]["y_rec"][0],
                ],
                "95% interval upper": [
                    u["intervals"]["x_pur"][1], u["intervals"]["y_pur"][1],
                    u["intervals"]["x_rec"][1], u["intervals"]["y_rec"][1],
                ],
                "Nominal inside interval": [
                    u["intervals"]["x_pur"][0] <= u["nominal"]["x_pur"] <= u["intervals"]["x_pur"][1],
                    u["intervals"]["y_pur"][0] <= u["nominal"]["y_pur"] <= u["intervals"]["y_pur"][1],
                    u["intervals"]["x_rec"][0] <= u["nominal"]["x_rec"] <= u["intervals"]["x_rec"][1],
                    u["intervals"]["y_rec"][0] <= u["nominal"]["y_rec"] <= u["intervals"]["y_rec"][1],
                ],
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

        units_df = pd.DataFrame({
            "Quantity group": ["Volumetric flow rates", "Concentrations", "Dimensionless quantities"],
            "Requirement": [
                "W_vol, F_vol, S_vol, S_min, and S_max must use one common unit.",
                "C_F, C_max, and all calculated concentrations must use one common unit.",
                "alpha_F, beta, D_Sc, D_Ex, abundances, recoveries, and transport factors are dimensionless."
            ]
        })
        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            units_df.to_excel(writer, sheet_name="Unit convention", index=False)
            input_df.to_excel(writer, sheet_name="Inputs", index=False)
            window_df.to_excel(writer, sheet_name="Operating window", index=False)
            stage_df.to_excel(writer, sheet_name="Stagewise profiles", index=False)
            iteration_df.to_excel(writer, sheet_name="Convergence History", index=False)
            if uncertainty_df is not None:
                uncertainty_df.to_excel(writer, sheet_name="Uncertainty propagation", index=False)

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
        return error_html, gr.update(), gr.update(), gr.update()

# ============================================================
# 9. Gradio UI Interface
# ============================================================
def clamp_m_value(n, m):
    try:
        n, m = int(n), int(m)
        if m > n: return gr.update(value=n)
        if m < 1: return gr.update(value=1)
        return gr.update()
    except Exception:
        return gr.update()

with gr.Blocks() as demo:
    gr.HTML(f"""
    <div class="center-title">
        <h1>Fractional Extraction Cascade Platform</h1>
        <p></p>
    </div>
    """)

    with gr.Tabs():
        # TAB 1: Forward Simulator
        with gr.TabItem("1. Steady-State Simulator (Forward)"):
            gr.HTML(get_schematic_html())
            with gr.Row():
                with gr.Column(scale=1, elem_classes="left-panel"):
                    gr.HTML("""
                    <div class='panel-card' style='background:#f8fafc; padding:10px 12px !important;'>
                        <div style='font-size:12.5px; color:#475569; line-height:1.55;'>
                            <b>Unit convention:</b> W<sub>vol</sub>, F<sub>vol</sub>, and S<sub>vol</sub> must use the same volumetric-flow unit.
                            C<sub>F</sub> and C<sub>max</sub> must use the same concentration unit.
                            The distribution ratios and separation factor are dimensionless.
                        </div>
                    </div>
                    """)
                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="section-heading">1. Cascade configuration</div>')
                        gr.HTML(make_field_label(f"Total number of theoretical stages, {var('N')}"))
                        N_input = gr.Number(value=50, show_label=False, precision=0)
                        gr.HTML(make_field_label(f"Feed stage, {var('M')}"))
                        M_input = gr.Number(value=30, show_label=False, precision=0)

                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="section-heading">2. Feed Parameters</div>')
                        gr.HTML(make_field_label(f"Total feed conc, {var('C', 'F')}"))
                        C_F_1 = gr.Number(value=0.1, show_label=False)
                        gr.HTML(make_field_label(f"Feed ratio, {var('α', 'F')} = {var('C', 'x,F')} / {var('C', 'y,F')}"))
                        alpha_F_1 = gr.Number(value=1.0, show_label=False)

                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="section-heading">3. Thermodynamic Parameters</div>')
                        gr.HTML(make_uncertainty_guide())
                        gr.HTML(make_field_label(f"Separation factor, {var('β')}"))
                        with gr.Row(equal_height=True):
                            beta_1 = gr.Number(value=1.02, show_label=False, scale=8, min_width=0)
                            gr.HTML("<div style='font-size:21px; font-weight:700; color:#475569; line-height:42px;'>±</div>", scale=1, min_width=32)
                            beta_err_1 = gr.Number(value=0.0, show_label=False, scale=7, min_width=0)
                        gr.HTML(make_field_label(f"Distribution ratio in scrubbing section, <i>D</i><sub>Sc</sub>"))
                        with gr.Row(equal_height=True):
                            D_Sc_1 = gr.Number(value=0.8, show_label=False, scale=8, min_width=0)
                            gr.HTML("<div style='font-size:21px; font-weight:700; color:#475569; line-height:42px;'>±</div>", scale=1, min_width=32)
                            D_Sc_err_1 = gr.Number(value=0.0, show_label=False, scale=7, min_width=0)
                        gr.HTML(make_field_label(f"Distribution ratio in extraction section, <i>D</i><sub>Ex</sub>"))
                        with gr.Row(equal_height=True):
                            D_Ex_1 = gr.Number(value=1.3, show_label=False, scale=8, min_width=0)
                            gr.HTML("<div style='font-size:21px; font-weight:700; color:#475569; line-height:42px;'>±</div>", scale=1, min_width=32)
                            D_Ex_err_1 = gr.Number(value=0.0, show_label=False, scale=7, min_width=0)
                        gr.HTML("<div class='field-note'>Thermodynamic means and SDs are dimensionless.</div>")

                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="section-heading">4. Hydrodynamic parameters</div>')
                        gr.HTML(make_field_label(f"Scrubbing phase flow rate, {var('W', 'vol')}"))
                        with gr.Row(equal_height=True):
                            W_vol_input = gr.Number(value=0.7, show_label=False, scale=8, min_width=0)
                            gr.HTML("<div style='font-size:21px; font-weight:700; color:#475569; line-height:42px;'>±</div>", scale=1, min_width=32)
                            W_err_1 = gr.Number(value=0.0, show_label=False, scale=7, min_width=0)
                        gr.HTML(make_field_label(f"Feed flow rate, {var('F', 'vol')}"))
                        with gr.Row(equal_height=True):
                            F_vol_input = gr.Number(value=0.4, show_label=False, scale=8, min_width=0)
                            gr.HTML("<div style='font-size:21px; font-weight:700; color:#475569; line-height:42px;'>±</div>", scale=1, min_width=32)
                            F_err_1 = gr.Number(value=0.0, show_label=False, scale=7, min_width=0)
                        gr.HTML(make_field_label(f"Organic phase flow rate, {var('S', 'vol')}"))
                        with gr.Row(equal_height=True):
                            S_vol_input = gr.Number(value=1.1, show_label=False, scale=8, min_width=0)
                            gr.HTML("<div style='font-size:21px; font-weight:700; color:#475569; line-height:42px;'>±</div>", scale=1, min_width=32)
                            S_err_1 = gr.Number(value=0.0, show_label=False, scale=7, min_width=0)
                        gr.HTML("<div class='field-note'>Each flow-rate SD uses the same unit as its mean.</div>")

                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="section-heading">5. Saturated load of organic phase</div>')
                        gr.HTML(make_field_label(f"Capacity limit, {var('C', 'max')}"))
                        C_max_1 = gr.Number(value=1.0, show_label=False)

                    run_btn = gr.Button("Run steady-state simulation", variant="primary", elem_classes="big-btn")

                with gr.Column(scale=2):
                    window_display = gr.HTML()
                    map_plot = gr.Plot(label="Operability map")
                    res_summary = gr.HTML("<div class='panel-card'>Click Run to simulate.</div>")
                    with gr.Row():
                        with gr.Column(scale=1):
                            prof_plot = gr.Plot(label="Steady-state cascade profiles")
                        with gr.Column(scale=1):
                            val_plot = gr.Plot(label="Numerical diagnostics")
                    file_dl = gr.File(label="Download Excel report")

        # TAB 2: Reverse Optimizer
        with gr.TabItem("2. Cascade Auto-Designer (Reverse)"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="left-panel"):
                    gr.HTML("""
                    <div class='panel-card' style='background:#f8fafc; padding:10px 12px !important;'>
                        <div style='font-size:12.5px; color:#475569; line-height:1.55;'>
                            <b>Unit convention:</b> all volumetric flow rates must share one unit, and C<sub>F</sub> and C<sub>max</sub> must share one concentration unit.
                            The optimized values are returned in the same units entered by the user.
                        </div>
                    </div>
                    """)
                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="section-heading">1. Optimization Target</div>')
                        gr.HTML("<p style='font-size:12px; color:#64748b;'>Select which isotopic abundance constraints the cascade must satisfy.</p>")
                        req_x_chk = gr.Checkbox(label="Target x Abundance (Organic Outlet)", value=True)
                        gr.HTML(make_field_label("x Abundance (%)"))
                        target_x_num = gr.Number(value=70.0, show_label=False)
                        req_y_chk = gr.Checkbox(label="Target y Abundance (Aqueous Outlet)", value=False)
                        gr.HTML(make_field_label("y Abundance (%)"))
                        target_y_num = gr.Number(value=80.0, show_label=False)

                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="section-heading">2. Feed Parameters</div>')
                        gr.HTML(make_field_label(f"Total feed conc, {var('C', 'F')}"))
                        C_F_2 = gr.Number(value=0.1, show_label=False)
                        gr.HTML(make_field_label(f"Feed ratio, {var('α', 'F')} = {var('C', 'x,F')} / {var('C', 'y,F')}"))
                        alpha_F_2 = gr.Number(value=1.0, show_label=False)

                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="section-heading">3. Thermodynamic Parameters</div>')
                        gr.HTML(make_uncertainty_guide(
                            f"In reverse design, {var('S', 'vol')} is optimized deterministically; "
                            f"no SD is assigned to or propagated for {var('S', 'vol')}."
                        ))
                        gr.HTML(make_field_label(f"Separation factor, {var('β')}"))
                        with gr.Row(equal_height=True):
                            beta_2 = gr.Number(value=1.02, show_label=False, scale=8, min_width=0)
                            gr.HTML("<div style='font-size:21px; font-weight:700; color:#475569; line-height:42px;'>±</div>", scale=1, min_width=32)
                            beta_err_2 = gr.Number(value=0.0, show_label=False, scale=7, min_width=0)
                        gr.HTML(make_field_label(f"Distribution ratio in scrubbing section, <i>D</i><sub>Sc</sub>"))
                        with gr.Row(equal_height=True):
                            D_Sc_2 = gr.Number(value=0.8, show_label=False, scale=8, min_width=0)
                            gr.HTML("<div style='font-size:21px; font-weight:700; color:#475569; line-height:42px;'>±</div>", scale=1, min_width=32)
                            D_Sc_err_2 = gr.Number(value=0.0, show_label=False, scale=7, min_width=0)
                        gr.HTML(make_field_label(f"Distribution ratio in extraction section, <i>D</i><sub>Ex</sub>"))
                        with gr.Row(equal_height=True):
                            D_Ex_2 = gr.Number(value=1.3, show_label=False, scale=8, min_width=0)
                            gr.HTML("<div style='font-size:21px; font-weight:700; color:#475569; line-height:42px;'>±</div>", scale=1, min_width=32)
                            D_Ex_err_2 = gr.Number(value=0.0, show_label=False, scale=7, min_width=0)
                        gr.HTML("<div class='field-note'>Thermodynamic means and SDs are dimensionless.</div>")

                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="section-heading">4. Industrial Constraints</div>')
                        gr.HTML("<p style='font-size:12px; color:#64748b;'>Set the allowable total stages based on your plant capacity.</p>")
                        gr.HTML(make_field_label(f"Allowed total stages, {var('N')}"))
                        max_n_limit_input = gr.Number(value=50, show_label=False, precision=0)
                        gr.HTML(make_field_label(f"Preset feed flow rate, {var('F', 'vol')}"))
                        with gr.Row(equal_height=True):
                            design_F_vol_input = gr.Number(value=0.4, show_label=False, scale=8, min_width=0)
                            gr.HTML("<div style='font-size:21px; font-weight:700; color:#475569; line-height:42px;'>±</div>", scale=1, min_width=32)
                            F_err_2 = gr.Number(value=0.0, show_label=False, scale=7, min_width=0)
                        gr.HTML("<div class='field-note'>The SD uses the same flow-rate unit as the mean.</div>")

                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="section-heading">5. Phase-flow variability</div>')
                        gr.HTML(make_field_label(f"Assumed SD of optimized {var('W', 'vol')}"))
                        W_err_2 = gr.Number(value=0.0, show_label=False)
                        gr.HTML(f"<div class='field-note'>Use the common flow-rate unit. SD = 0 keeps {var('W', 'vol')} fixed.</div>")

                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="section-heading">6. Saturated load of organic phase</div>')
                        gr.HTML(make_field_label(f"Capacity limit, {var('C', 'max')}"))
                        C_max_2 = gr.Number(value=1.0, show_label=False)
                        
                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="section-heading">Launch Architect</div>')
                        gr.HTML(f"""<p style='font-size:12px; color:#64748b;'>The program keeps {var('N')} and {var('F', 'vol')} fixed, then searches {var('M')}, {var('W', 'vol')}, and {var('S', 'vol')}.</p>""")
                        gr.HTML(make_field_label("Search mode"))
                        search_mode_input = gr.Radio(choices=["Fast", "Thorough"], value="Fast", show_label=False)
                        opt_btn = gr.Button("Find Optimal Architecture", variant="primary", elem_classes="big-btn")

                with gr.Column(scale=2):
                    opt_summary = gr.HTML("""
                    <div class="panel-card">
                        <h3 style="color:#64748b; font-size:16px;">
                            Click 'Find Optimal Architecture' to launch the Heuristic Grid-Search. <br><br>
                            <span style="font-size:13px; font-weight:normal;">
                            The engine employs a <b>Coarse-to-Fine Heuristic Grid</b>, rejects candidates with less than 95% operating-window probability when uncertainty is active, and uncertainty-validates the best-ranked finalists.
                            </span>
                        </h3>
                    </div>
                    """)
                    opt_file_dl = gr.File(label="Download Excel report for optimized condition")

    gr.HTML("""
    <div class="app-footer">
        <div>
            <b>Developed by:</b>
            Dr. Qi ZHAO (HKU), Kaimin SHIH (HKU), Dr. Wei SHEN (LUH), Dr. Junqiang YANG (LZH), and Mr. Junyi Wang (HKU).
        </div>
        <div>
            For inquiries or suggestions regarding this project, please feel free to contact
            <a href="mailto:zhaoqi22@hku.hk">zhaoqi22@hku.hk</a> (Q. ZHAO) or
            <a href="mailto:kshih@hku.hk">kshih@hku.hk</a> (K. SHIH).
        </div>
    </div>
    """)

    # Events
    N_input.change(fn=clamp_m_value, inputs=[N_input, M_input], outputs=[M_input])
    dynamic_inputs = [
        W_vol_input, F_vol_input, S_vol_input, alpha_F_1,
        beta_1, D_Sc_1, D_Ex_1,
        beta_err_1, D_Sc_err_1, D_Ex_err_1,
        W_err_1, F_err_1, S_err_1,
    ]
    
    for inp in dynamic_inputs:
        inp.change(fn=update_window_outputs, inputs=dynamic_inputs, outputs=[window_display, map_plot])
    demo.load(fn=update_window_outputs, inputs=dynamic_inputs, outputs=[window_display, map_plot])

    run_btn.click(
        fn=solve_fractional_extraction,
        inputs=[
            N_input, M_input, W_vol_input, F_vol_input, S_vol_input,
            C_F_1, alpha_F_1, beta_1, D_Sc_1, D_Ex_1, C_max_1,
            beta_err_1, D_Sc_err_1, D_Ex_err_1, W_err_1, F_err_1, S_err_1,
        ],
        outputs=[res_summary, prof_plot, val_plot, file_dl]
    )

    opt_btn.click(
        fn=run_optimizer,
        inputs=[
            req_x_chk, req_y_chk, target_x_num, target_y_num,
            max_n_limit_input, design_F_vol_input, search_mode_input,
            C_F_2, alpha_F_2, beta_2, D_Sc_2, D_Ex_2, C_max_2,
            beta_err_2, D_Sc_err_2, D_Ex_err_2, W_err_2, F_err_2,
        ],
        outputs=[opt_summary, opt_file_dl]
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(),
        css=custom_css,
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        inbrowser=False,
    )
