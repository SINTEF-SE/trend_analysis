import numpy as np
from scipy.stats import chi2
import math

# -------------------------
# Helpers: time indexing
# -------------------------

def monthly_indices(df):
    """Return integer month indices starting at 0 for the earliest row."""
    months = df["date"].dt.month + df["date"].dt.year * 12
    months = months - months.min()
    return months

def monthly_series(df):
    """Return a dense monthly count series y (length = max_month+1)."""
    m = monthly_indices(df)
    L = int(m.max()) + 1 if len(m) else 0
    y = np.zeros(L, dtype=int)
    if L > 0:
        # bincount length trick: minlength = max_index+1
        y = np.bincount(m, minlength=L)
    return y

# -------------------------
# Helpers: disjoint binning with phase
# -------------------------

def bin_counts_with_phase(y, months_window, phase):
    """
    Disjoint, phase-shifted binning of a dense monthly series y.
    phase is an integer in [0, months_window-1].
    Returns counts per complete bin (drops any trailing partial bin).
    """
    if len(y) == 0:
        return np.array([], dtype=int)

    # Start at 'phase'; slice y[phase:] then take windows of length months_window
    ys = y[phase:]
    B = len(ys) // months_window  # complete bins only
    if B <= 0:
        return np.array([], dtype=int)

    trimmed = ys[:B * months_window]
    counts = trimmed.reshape(B, months_window).sum(axis=1)
    return counts

def phase_offsets(months_window, K=5):
    """
    Choose up to K phase offsets evenly spaced in [0, months_window-1].
    If months_window < K, use all unique phases.
    """
    K_eff = min(K, months_window)
    # Evenly spaced remainders
    #offsets = np.linspace(0, months_window-1, num=K_eff, dtype=int)
    offsets = np.rint(np.linspace(0, months_window, num=K_eff+1)[:-1]).astype(int)
    return np.unique(offsets)

# -------------------------
# Helpers: combine p values
# -------------------------

def simes_combine_pvalues(pvals):
    p = np.array([x for x in pvals if np.isfinite(x)])
    if len(p) == 0:
        return np.nan
    p.sort()
    K = len(p)
    return min(1.0, np.min(K * p / np.arange(1, K+1)))

# -------------------------
# Metrics
# -------------------------

def reduced_chi2(counts):
    """
    Reduced chi-square for uniformity over counts vector.
    Returns (chi2_reduced, p) with df = B-1.
    """
    B = len(counts)
    if B < 2:
        return np.nan, np.nan
    N = counts.sum()
    if N == 0:
        return np.nan, np.nan

    mean = N / B
    # X^2 = sum (c_i - mean)^2 / mean
    X2 = ((counts - mean) ** 2 / mean).sum()
    df = B - 1
    chi2_reduced = X2 / df
    p = chi2.sf(X2, df)  # right tail
    return chi2_reduced, p

# -------------------------
# Main plotting function
# -------------------------

def calc_nonuniformity(node,
                     months_window_range=(6, 24), # including ends
                     phase_K=5,
                     min_bins=2):

    skipped_all = True

    node.nonuniformity_scores = {}
    node.nonuniformity_plot_x = {}
    node.nonuniformity_plot_y = {}


    for key in node.dfs:
        df = node.dfs[key]
        if len(df) == 0:
            continue

        # ensure at least 5 expected values per bin, so statistics stay clean
        number_of_months = max(monthly_indices(df))
        min_window_size = 5 * number_of_months/len(df)
        min_window_size = math.ceil(min_window_size)
        if min_window_size >= months_window_range[1]: # need at least 2 possible values
            continue
        min_window_size = min(min_window_size, months_window_range[0])
        clipped_window_range = range(min_window_size, months_window_range[1]+1) # +1 to include end

        # Dense monthly series for wavelet and for fast binning
        y = monthly_series(df)
        if len(y) == 0:
            continue

        # 1) Phase-averaged reduced chi-square
        X_sizes = []
        chi_phase_avg = []

        # 2) Sidák-corrected p across phases (per window size)
        p_simes_list = []


        for months_window in clipped_window_range:
            phases = phase_offsets(months_window, K=phase_K)

            chi_vals = []
            p_vals = []

            for ph in phases:
                counts = bin_counts_with_phase(y, months_window, ph)
                if len(counts) < min_bins:
                    continue
                
                # Guard: expected count per bin
                mean_count = counts.mean()
                if mean_count < 5:
                    # Skip this phase for this window size if too sparse
                    continue

                chi_r, p_r = reduced_chi2(counts)
                if np.isfinite(chi_r):
                    chi_vals.append(chi_r)
                if np.isfinite(p_r):
                    p_vals.append(p_r)


            if len(chi_vals) == 0 or len(p_vals) == 0:
                # Not enough phases produced valid stats for this window size
                continue

            # Phase-robust summaries
            chi_med = np.median(chi_vals)              # phase-averaged reduced χ²

            X_sizes.append(months_window)
            chi_phase_avg.append(chi_med)
            p_simes_list.append(simes_combine_pvalues(p_vals))

        # Plot: phase-averaged reduced chi-square
        if len(X_sizes):
            node.nonuniformity_plot_x[key] = X_sizes

            p_arr = np.clip(np.array(p_simes_list), 1e-50, 1.0)
            node.nonuniformity_plot_y[key] = -np.log10(p_arr)

            node.nonuniformity_scores[key] = simes_combine_pvalues(p_arr)

            skipped_all = False


    
    if skipped_all: 
        return # stop traversing this branch, since no child node will have enough data either
    else:
        for child in node.children:
            calc_nonuniformity(child,
                     months_window_range,
                     phase_K,
                     min_bins)
