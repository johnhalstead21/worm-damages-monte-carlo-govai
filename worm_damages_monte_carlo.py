"""
Monte Carlo simulation of expected damages from at least one data-damaging worm attack.

Compares:
  - Baseline scenario
  - Conditional on Capability 1 (AI uplift scenario)

For each estimator (Author, Experts, Forecasters), samples from fitted distributions
for end-to-end capability, willingness, and potential damages.  Computes:
  P(at least one worm) = 1 - prod(1 - cap_i * will_i)  across 5 threat actors
  Expected damages     = P(at least one worm) * damages_per_event

Two distribution choices for [0,1]-bounded parameters (capability, willingness):
  (1) Lognormal, clipped to [0,1]
  (2) Beta distribution, which naturally respects [0,1] bounds

Damages per event always use a lognormal (unbounded positive).

Data source: 'Combined author and survey' tab of the worm damages spreadsheet (v4).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import warnings
from scipy import stats as sp_stats
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
np.random.seed(42)
N_SAMPLES = 200_000

# ============================================================
# DATA from 'Combined author and survey' tab (corrected v4)
# Format: (5th_percentile, median, 95th_percentile)
# ============================================================

# --- BASELINE end-to-end capability (rows 92-96) ---
baseline_cap = {
    'Author': {
        'TA1': (0, 0, 0),
        'TA2': (1e-6, 0.0001005, 0.0002),
        'TA3': (0.0005, 0.03025, 0.06),
        'TA4': (0.05, 0.325, 0.6),
        'TA5': (0.9, 0.945, 0.99),
    },
    'Experts': {
        'TA1': (0, 0, 0),
        'TA2': (0, 1.1e-7, 1.4e-5),
        'TA3': (2e-5, 0.0005775, 0.0025),
        'TA4': (0.0045, 0.1485, 0.36834),
        'TA5': (0.165, 0.9212, 0.9748545),       # corrected p5
    },
    'Forecasters': {
        'TA1': (6e-8, 1e-5, 0.0004),
        'TA2': (6e-6, 0.00035, 0.0144),
        'TA3': (0.006, 0.125, 0.195),
        'TA4': (0.4, 0.665, 0.776),
        'TA5': (0.855, 0.9405, 0.99),
    },
}

# --- CONDITIONAL ON CAPABILITY 1: end-to-end capability (rows 144-148) ---
cond_cap = {
    'Author': {
        'TA1': (0.00025, 0.002625, 0.005),
        'TA2': (0.1, 0.2, 0.3),
        'TA3': (0.855, 0.893, 0.931),
        'TA4': (0.99, 0.995, 1.0),
        'TA5': (0.99, 0.995, 1.0),
    },
    'Experts': {
        'TA1': (1e-5, 0.0005075, 0.0032),
        'TA2': (0, 0.0224775, 0.1225),
        'TA3': (0.0005, 0.12375, 0.8084),
        'TA4': (0.006, 0.813875, 0.96528125),
        'TA5': (0.285, 0.98902625, 0.99),
    },
    'Forecasters': {
        'TA1': (1e-5, 0.00075, 0.0192),
        'TA2': (0.002, 0.064, 0.2),
        'TA3': (0.21, 0.39, 0.64),
        'TA4': (0.738, 0.882, 0.97),
        'TA5': (0.931, 0.9801, 0.99),
    },
}

# --- Willingness (rows 101-105, same for both scenarios) ---
willingness = {
    'Author': {
        'TA1': (0.98, 0.99, 1.0),
        'TA2': (0.7, 0.85, 1.0),             # corrected p5
        'TA3': (0.02, 0.035, 0.05),
        'TA4': (0.02, 0.035, 0.05),
        'TA5': (0.01, 0.03, 0.05),
    },
    'Experts': {
        'TA1': (0.0, 0.8998, 1.0),
        'TA2': (0.0, 0.8475, 0.9975),
        'TA3': (0.1, 0.6, 0.9),
        'TA4': (0.215, 0.655, 0.8425),
        'TA5': (0.015, 0.35, 0.64),
    },
    'Forecasters': {
        'TA1': (0.9, 0.985, 1.0),
        'TA2': (0.9, 0.985, 0.99),
        'TA3': (0.62, 0.9, 0.95),
        'TA4': (0.08, 0.18, 0.39),
        'TA5': (0.04, 0.05, 0.15),
    },
}

# --- Potential damages per event ---
damages_triplet = (10e9, 31.622776602e9, 100e9)  # ($10B, ~$31.6B geomean, $100B)

# ============================================================
# Distribution fitting utilities
# ============================================================

def fit_lognormal(p5, p50, p95):
    """Fit lognormal (mu, sigma) to 5th/50th/95th percentiles."""
    if p50 <= 0:
        return None
    mu = np.log(p50)
    if p95 > p5 and p5 > 0:
        sigma = ((np.log(p95) - mu) / 1.6449 + (mu - np.log(p5)) / 1.6449) / 2
    elif p95 > p50:
        sigma = (np.log(p95) - mu) / 1.6449
    elif p5 > 0:
        sigma = (mu - np.log(p5)) / 1.6449
    else:
        sigma = 0.1
    return (mu, max(sigma, 0.01))


def fit_beta(p5, p50, p95):
    """
    Fit a Beta(a, b) distribution to 5th/50th/95th percentile constraints.
    Uses least-squares optimisation over a grid of starting points.
    """
    def objective(params):
        a, b = params
        if a <= 0 or b <= 0:
            return 1e10
        try:
            q5 = sp_stats.beta.ppf(0.05, a, b)
            q50 = sp_stats.beta.ppf(0.50, a, b)
            q95 = sp_stats.beta.ppf(0.95, a, b)
            return (q5 - p5)**2 + (q50 - p50)**2 + (q95 - p95)**2
        except Exception:
            return 1e10

    best = None
    for a0 in [0.3, 0.5, 1, 2, 5, 10, 20, 50]:
        for b0 in [0.3, 0.5, 1, 2, 5, 10, 20, 50, 100]:
            res = minimize(objective, [a0, b0], method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-14})
            if best is None or res.fun < best.fun:
                best = res
    return best.x


def sample_lognormal(low, med, high, n, clip_min=None, clip_max=None):
    """
    Draw n samples from a lognormal fitted to (5th, 50th, 95th) percentiles.
    Handles edge cases: all-zero, zero-median-with-nonzero-tail, zero-low-bound.
    """
    if low == med == high:
        return np.full(n, med)
    if med == 0 and high == 0:
        return np.zeros(n)
    if med == 0 and high > 0:
        # Median is 0 => more than half the mass at 0
        mask = np.random.random(n) > 0.6
        samples = np.zeros(n)
        if mask.sum() > 0:
            params = fit_lognormal(max(high * 0.01, 1e-15), high * 0.1, high)
            if params:
                samples[mask] = np.random.lognormal(params[0], params[1], mask.sum())
        return np.clip(samples, 0, clip_max) if clip_max else samples
    if low == 0 and med > 0:
        # Small fraction are zero, rest follow lognormal
        mask = np.random.random(n) > 0.03
        samples = np.zeros(n)
        if mask.sum() > 0:
            params = fit_lognormal(med * 0.05, med, high)
            if params:
                samples[mask] = np.random.lognormal(params[0], params[1], mask.sum())
        return np.clip(samples, 0, clip_max) if clip_max else samples
    params = fit_lognormal(max(low, 1e-15), med, high)
    samples = np.random.lognormal(params[0], params[1], n) if params else np.full(n, med)
    if clip_min is not None:
        samples = np.clip(samples, clip_min, None)
    if clip_max is not None:
        samples = np.clip(samples, None, clip_max)
    return samples


def sample_beta(low, med, high, n):
    """
    Draw n samples from a Beta distribution fitted to (5th, 50th, 95th) percentiles.
    Naturally bounded to [0,1]. Handles edge cases.
    """
    if low == med == high:
        return np.full(n, med)
    if med == 0 and high == 0:
        return np.zeros(n)
    if med == 0 and high > 0:
        # Median is 0 => more than half the mass at 0; use mixture
        mask = np.random.random(n) > 0.6
        samples = np.zeros(n)
        if mask.sum() > 0:
            a, b = fit_beta(max(high * 0.01, 1e-6), high * 0.1, min(high, 0.999))
            samples[mask] = sp_stats.beta.rvs(a, b, size=mask.sum())
        return samples
    if low == 0 and med > 0:
        # Small fraction are zero, rest follow beta
        mask = np.random.random(n) > 0.03
        samples = np.zeros(n)
        if mask.sum() > 0:
            a, b = fit_beta(med * 0.05, med, min(high, 0.999))
            samples[mask] = sp_stats.beta.rvs(a, b, size=mask.sum())
        return samples
    # Standard case
    a, b_param = fit_beta(max(low, 1e-6), med, min(high, 0.9999))
    return sp_stats.beta.rvs(a, b_param, size=n)


def sample_bounded(low, med, high, n, method='lognormal'):
    """Sample from a [0,1]-bounded parameter using the specified method."""
    if method == 'beta':
        return sample_beta(low, med, high, n)
    else:
        return sample_lognormal(low, med, high, n, clip_min=0, clip_max=1)


# ============================================================
# Core Monte Carlo engine
# ============================================================

def compute_damages_mc(cap_data, will_data, damages_tri, n=N_SAMPLES, method='lognormal'):
    """
    Monte Carlo: expected damages from >= 1 data-damaging worm.
    P(>=1 worm) = 1 - prod_i(1 - capability_i * willingness_i)
    Expected damages = P(>=1) * damages_per_event
    """
    dmg = sample_lognormal(damages_tri[0], damages_tri[1], damages_tri[2], n)
    surv = np.ones(n)
    for ta in ['TA1', 'TA2', 'TA3', 'TA4', 'TA5']:
        c = sample_bounded(*cap_data[ta], n, method=method)
        w = sample_bounded(*will_data[ta], n, method=method)
        surv *= (1 - np.clip(c * w, 0, 1))
    return (1 - surv) * dmg


def compute_ta_proportions(cap_data, will_data, damages_tri, n=N_SAMPLES, method='lognormal'):
    """
    Compute per-sample TA contribution proportions using additive model.
    Returns median proportion for each TA (normalised to sum to 1).
    """
    dmg = sample_lognormal(damages_tri[0], damages_tri[1], damages_tri[2], n)
    ta_dmg = {}
    for ta in ['TA1', 'TA2', 'TA3', 'TA4', 'TA5']:
        c = sample_bounded(*cap_data[ta], n, method=method)
        w = sample_bounded(*will_data[ta], n, method=method)
        ta_dmg[ta] = c * w * dmg
    total = sum(ta_dmg[ta] for ta in ['TA1', 'TA2', 'TA3', 'TA4', 'TA5'])
    props = {}
    for ta in ['TA1', 'TA2', 'TA3', 'TA4', 'TA5']:
        with np.errstate(divide='ignore', invalid='ignore'):
            p = np.where(total > 0, ta_dmg[ta] / total, 0)
        props[ta] = np.median(p)
    ps = sum(props.values())
    if ps > 0:
        props = {ta: v / ps for ta, v in props.items()}
    return props


# ============================================================
# Formatting
# ============================================================

def fmt(val):
    if val >= 1e12:
        return f"${val/1e12:.0f}T"
    if val >= 1e9:
        v = val / 1e9
        return f"${v:.0f}B" if v >= 10 else f"${v:.1f}B"
    if val >= 1e6:
        v = val / 1e6
        return f"${v:.0f}M" if v >= 10 else f"${v:.1f}M"
    if val >= 1e3:
        return f"${val/1e3:.0f}K"
    return f"${val:.0f}"


def dollar_formatter(val, pos):
    if val >= 1e12:
        return f"${val/1e12:.0f}T"
    if val >= 1e9:
        return f"${val/1e9:.0f}B"
    if val >= 1e6:
        return f"${val/1e6:.0f}M"
    if val == 0:
        return "$0"
    return f"${val:.0f}"


# ============================================================
# Chart styling constants
# ============================================================

BG_COLOR = '#F0EDE5'
COLORS = {'Author': '#D4650B', 'Experts': '#F0A030', 'Forecasters': '#F5D08E'}
TA_COLORS = ['#FADBD8', '#F1948A', '#E74C3C', '#B71C1C', '#4A0000']
TA_LABELS = ['TA1: Hobbyist', 'TA2: Professional', 'TA3: Team of 10',
             'TA4: 100 state-level', 'TA5: 1,000 state-level']
ESTIMATORS = ['Author', 'Experts', 'Forecasters']


# ============================================================
# Chart 1: Baseline vs Conditional (log + linear)
# ============================================================

def make_main_chart(all_stats, log_scale, filename, method_label=''):
    """Box-plot chart comparing baseline and conditional scenarios."""
    bar_w = 0.45
    bar_gap = 0.10
    group_gap = 0.4
    groups = ['baseline', 'conditional']
    group_labels = ['Baseline', 'AI elite exploit\ncapability uplift']
    group_starts = [0, 3 * (bar_w + bar_gap) + group_gap]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    for gi, group in enumerate(groups):
        for ei, est in enumerate(ESTIMATORS):
            s = all_stats[f'{est}_{group}']
            x = group_starts[gi] + ei * (bar_w + bar_gap)
            p5_raw = s['p5']
            p95 = s['p95']
            med = s['median']

            if log_scale:
                VISUAL_FLOOR = 1e8
                p5_vis = max(p5_raw, VISUAL_FLOOR)
                rect = plt.Rectangle((x, p5_vis), bar_w, p95 - p5_vis,
                                      facecolor=COLORS[est], edgecolor='none', zorder=2)
                ax.add_patch(rect)
                ax.text(x + bar_w / 2, p95 * 1.2, fmt(p95),
                        ha='center', va='bottom', fontsize=7.5, color='#555555', zorder=5)
                label_5 = fmt(p5_raw) if p5_raw >= 1e6 else "$0"
                ax.text(x + bar_w / 2, p5_vis * 0.6, label_5,
                        ha='center', va='top', fontsize=7.5, color='#555555', zorder=5)
            else:
                p5 = p5_raw
                rect = plt.Rectangle((x, p5), bar_w, p95 - p5,
                                      facecolor=COLORS[est], edgecolor='none', zorder=2)
                ax.add_patch(rect)
                ax.text(x + bar_w / 2, p95 + 1.5e9, fmt(p95),
                        ha='center', va='bottom', fontsize=7.5, color='#555555', zorder=5)
                label_5 = fmt(p5) if p5 >= 1e6 else "$0"
                ax.text(x + bar_w / 2, max(p5 - 1.5e9, -3e9), label_5,
                        ha='center', va='top', fontsize=7.5, color='#555555', zorder=5)

            ax.plot([x - 0.03, x + bar_w + 0.03], [med, med],
                    color='black', linewidth=2.5, zorder=4, solid_capstyle='round')
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor="black", edgecolor="none")
            ax.text(x + bar_w / 2, med, fmt(med),
                    ha='center', va='center', fontsize=8.5, fontweight='bold',
                    color='white', bbox=bbox_props, zorder=5)

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(3e7, 2e11)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    else:
        ax.set_ylim(-5e9, 95e9)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10e9))

    ax.set_xlim(-0.6, group_starts[1] + 3 * (bar_w + bar_gap) + 0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('#cccccc')
    ax.tick_params(left=True, bottom=False, labelbottom=False, labelleft=True,
                   colors='#777777', which='both')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
    ax.tick_params(axis='y', labelsize=8.5, labelcolor='#555555')
    ax.grid(axis='y', which='major', color='#dddddd', linewidth=0.5, zorder=0)

    for gi, label in enumerate(group_labels):
        center_x = group_starts[gi] + 1 * (bar_w + bar_gap) + bar_w / 2
        y_label = 2.2e7 if log_scale else -7e9
        ax.text(center_x, y_label, label,
                ha='center', va='top', fontsize=11, fontweight='bold', color='#333333')

    legend_elems = [
        mpatches.Patch(facecolor=COLORS[e], edgecolor='none', label=e) for e in ESTIMATORS
    ]
    ax.legend(handles=legend_elems, loc='upper left',
              frameon=True, framealpha=0.95, facecolor='white',
              edgecolor='#dddddd', fontsize=8.5, title='Estimates by:',
              title_fontsize=9, bbox_to_anchor=(0.01, 0.99))

    title = 'Expected damages from at least one\ndata-damaging worm attack'
    if method_label:
        title += f' ({method_label})'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98, color='#222222')
    ax.set_ylabel('Expected damages', fontsize=10, color='#555555', labelpad=8)

    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    for ext in ['png', 'svg']:
        plt.savefig(f'{filename}.{ext}', dpi=200, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {filename}.png/.svg")


# ============================================================
# Chart 2: Marginal damages (simple subtraction of statistics)
# ============================================================

def make_marginal_chart(all_stats, log_scale, filename, method_label=''):
    """Marginal damages = conditional stats minus baseline stats."""
    marginal = {}
    for est in ESTIMATORS:
        bl = all_stats[f'{est}_baseline']
        cd = all_stats[f'{est}_conditional']
        marginal[est] = {
            'p5': cd['p5'] - bl['p5'],
            'median': cd['median'] - bl['median'],
            'p95': cd['p95'] - bl['p95'],
        }

    bar_w = 0.55
    bar_gap = 0.25
    total_w = 3 * (bar_w + bar_gap) - bar_gap

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    for ei, est in enumerate(ESTIMATORS):
        s = marginal[est]
        x = ei * (bar_w + bar_gap)
        p5 = s['p5']
        p95 = s['p95']
        med = s['median']

        if log_scale:
            p5v = max(p5, 1e8)
        else:
            p5v = p5

        rect = plt.Rectangle((x, p5v), bar_w, p95 - p5v,
                              facecolor=COLORS[est], edgecolor='none', zorder=2)
        ax.add_patch(rect)

        ax.plot([x - 0.03, x + bar_w + 0.03], [med, med], color='black',
                linewidth=2.5, zorder=4, solid_capstyle='round')
        ax.text(x + bar_w / 2, med, fmt(med), ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", edgecolor="none"))

        if log_scale:
            ax.text(x + bar_w / 2, p95 * 1.2, fmt(p95), ha='center', va='bottom',
                    fontsize=8, color='#555555', zorder=5)
            ax.text(x + bar_w / 2, p5v * 0.55, fmt(p5) if p5 >= 1e6 else "$0",
                    ha='center', va='top', fontsize=8, color='#555555', zorder=5)
        else:
            ax.text(x + bar_w / 2, p95 + 0.8e9, fmt(p95), ha='center', va='bottom',
                    fontsize=8, color='#555555', zorder=5)
            ax.text(x + bar_w / 2, p5 - 0.8e9, fmt(p5) if p5 >= 1e6 else "$0",
                    ha='center', va='top', fontsize=8, color='#555555', zorder=5)

    ax.set_xlim(-0.3, total_w + 0.3)
    ax.set_xticks([i * (bar_w + bar_gap) + bar_w / 2 for i in range(3)])
    ax.set_xticklabels(ESTIMATORS, fontsize=9)

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(5e7, 5e10)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    else:
        ax.set_ylim(-2e9, 22e9)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5e9))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('#cccccc')
    ax.tick_params(left=True, bottom=False, colors='#777777', which='both')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
    ax.tick_params(axis='y', labelsize=8.5, labelcolor='#555555')
    ax.grid(axis='y', which='major', color='#dddddd', linewidth=0.5, zorder=0)
    ax.set_ylabel('Expected damages', fontsize=10, color='#555555', labelpad=8)
    ax.set_xlabel('Marginal damages from AI elite exploit capability uplift',
                  fontsize=10, fontweight='bold', color='#333333', labelpad=12)

    title = 'Marginal expected damages from\nAI elite exploit capability uplift'
    if method_label:
        title += f' ({method_label})'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98, color='#222222')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ['png', 'svg']:
        plt.savefig(f'{filename}.{ext}', dpi=200, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {filename}.png/.svg")


# ============================================================
# Chart 3: TA shares (100% stacked bar)
# ============================================================

def make_ta_shares_chart(props_all, filename, method_label=''):
    """Stacked bar chart showing per-TA share of expected damages."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5.5), sharey=True)
    fig.patch.set_facecolor(BG_COLOR)

    bar_w = 0.55
    scenarios = [
        ('baseline', 'Baseline', ax1),
        ('conditional', 'AI elite exploit\ncapability uplift', ax2),
    ]

    for scenario, title, ax in scenarios:
        ax.set_facecolor(BG_COLOR)
        for ei, est in enumerate(ESTIMATORS):
            p = props_all[f'{est}_{scenario}']
            bottom = 0
            for ti, ta in enumerate(['TA1', 'TA2', 'TA3', 'TA4', 'TA5']):
                val = p[ta] * 100
                ax.bar(ei, val, bar_w, bottom=bottom, color=TA_COLORS[ti],
                       edgecolor='white', linewidth=0.8, zorder=2)
                if val >= 5:
                    txt_color = 'white' if ti >= 2 else 'black'
                    ax.text(ei, bottom + val / 2, f"{val:.0f}%",
                            ha='center', va='center', fontsize=8, fontweight='bold',
                            color=txt_color, zorder=3)
                bottom += val

        ax.set_xticks(range(3))
        ax.set_xticklabels(ESTIMATORS, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold', color='#333333', pad=12)
        ax.set_ylim(0, 105)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.tick_params(bottom=False, colors='#777777')
        ax.grid(axis='y', color='#dddddd', linewidth=0.5, zorder=0)
        ax.tick_params(axis='y', labelsize=8.5, labelcolor='#555555')

    ax1.set_ylabel('Share of median expected damages', fontsize=10, color='#555555', labelpad=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.0f}%"))

    ta_legend = [mpatches.Patch(facecolor=TA_COLORS[i], edgecolor='#ccc', linewidth=0.5,
                 label=TA_LABELS[i]) for i in range(5)]
    fig.legend(handles=ta_legend, loc='lower center', ncol=3,
               frameon=True, framealpha=0.95, facecolor='white',
               edgecolor='#dddddd', fontsize=7.5, title='Threat actors (light \u2192 dark):',
               title_fontsize=8.5, bbox_to_anchor=(0.5, -0.02))

    title = 'Share of expected damages by threat actor'
    if method_label:
        title += f' ({method_label})'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02, color='#222222')

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    for ext in ['png', 'svg']:
        plt.savefig(f'{filename}.{ext}', dpi=200, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {filename}.png/.svg")


# ============================================================
# Run everything for a given method
# ============================================================

def run_analysis(method, file_prefix):
    """Run full MC analysis and generate all charts for a distribution method."""
    label = 'lognormal' if method == 'lognormal' else 'beta'
    print(f"\n{'='*60}")
    print(f"  Running Monte Carlo with {label} distributions")
    print(f"  (N = {N_SAMPLES:,} samples)")
    print(f"{'='*60}")

    # Reset seed for reproducibility
    np.random.seed(42)

    # --- Main simulation ---
    all_stats = {}
    for est in ESTIMATORS:
        bl = compute_damages_mc(baseline_cap[est], willingness[est], damages_triplet,
                                method=method)
        cd = compute_damages_mc(cond_cap[est], willingness[est], damages_triplet,
                                method=method)
        for lbl, arr in [('baseline', bl), ('conditional', cd)]:
            all_stats[f'{est}_{lbl}'] = {
                'p5': np.percentile(arr, 5),
                'median': np.percentile(arr, 50),
                'p95': np.percentile(arr, 95),
            }

    # Print summary
    print(f"\n{'Scenario':<14} {'Estimator':<14} {'5th %ile':>12} {'Median':>12} {'95th %ile':>12}")
    print("-" * 68)
    for group in ['baseline', 'conditional']:
        for est in ESTIMATORS:
            s = all_stats[f'{est}_{group}']
            print(f"{group:<14} {est:<14} {fmt(s['p5']):>12} {fmt(s['median']):>12} {fmt(s['p95']):>12}")

    # Verification against spreadsheet medians
    print(f"\n--- Verification (MC median vs spreadsheet) ---")
    ss_baseline = {'Author': 1280770629, 'Experts': 12286666646, 'Forecasters': 8087107766}
    ss_cond = {'Author': 7876122470, 'Experts': 22859847192, 'Forecasters': 16250331804}
    for est in ESTIMATORS:
        mc_bl = all_stats[f'{est}_baseline']['median']
        mc_cd = all_stats[f'{est}_conditional']['median']
        print(f"  {est} baseline:    MC={fmt(mc_bl):>8}, Sheet={fmt(ss_baseline[est]):>8}, ratio={mc_bl/ss_baseline[est]:.2f}")
        print(f"  {est} conditional: MC={fmt(mc_cd):>8}, Sheet={fmt(ss_cond[est]):>8}, ratio={mc_cd/ss_cond[est]:.2f}")

    # Marginal summary
    print(f"\n--- Marginal damages (conditional - baseline) ---")
    for est in ESTIMATORS:
        bl = all_stats[f'{est}_baseline']
        cd = all_stats[f'{est}_conditional']
        print(f"  {est}: p5={fmt(cd['p5']-bl['p5'])}, median={fmt(cd['median']-bl['median'])}, p95={fmt(cd['p95']-bl['p95'])}")

    # --- TA proportions ---
    np.random.seed(42)
    props_all = {}
    for scenario, cap_data in [('baseline', baseline_cap), ('conditional', cond_cap)]:
        for est in ESTIMATORS:
            props_all[f'{est}_{scenario}'] = compute_ta_proportions(
                cap_data[est], willingness[est], damages_triplet, method=method)

    print(f"\n--- TA share proportions ---")
    for scenario in ['baseline', 'conditional']:
        print(f"  {scenario.upper()}:")
        for est in ESTIMATORS:
            p = props_all[f'{est}_{scenario}']
            parts = [f"{ta}: {p[ta]*100:.1f}%" for ta in ['TA1', 'TA2', 'TA3', 'TA4', 'TA5']]
            print(f"    {est}: {', '.join(parts)}")

    # --- Generate charts ---
    print(f"\nGenerating charts...")
    method_label = label

    make_main_chart(all_stats, log_scale=True,
                    filename=f'{file_prefix}_chart',
                    method_label=method_label)
    make_main_chart(all_stats, log_scale=False,
                    filename=f'{file_prefix}_chart_linear',
                    method_label=method_label)
    make_marginal_chart(all_stats, log_scale=True,
                        filename=f'{file_prefix}_marginal_log',
                        method_label=method_label)
    make_marginal_chart(all_stats, log_scale=False,
                        filename=f'{file_prefix}_marginal_linear',
                        method_label=method_label)
    make_ta_shares_chart(props_all,
                         filename=f'{file_prefix}_ta_shares',
                         method_label=method_label)

    return all_stats, props_all


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import os
    outdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(outdir)

    ln_stats, ln_props = run_analysis('lognormal', 'worm_lognormal')
    beta_stats, beta_props = run_analysis('beta', 'worm_beta')

    print("\n" + "=" * 60)
    print("  All charts generated successfully!")
    print("=" * 60)
