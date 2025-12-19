from metrics.nonuniformity import simes_combine_pvalues

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

# -------------------------
# Helpers
# -------------------------

def monthly_indices(df):
    months = df["date"].dt.month + df["date"].dt.year * 12
    months = months - months.min()
    return months

def monthly_series(df):
    m = monthly_indices(df)
    L = int(m.max()) + 1 if len(m) else 0
    y = np.zeros(L, dtype=int)
    if L > 0:
        y = np.bincount(m, minlength=L)
    return y

def bin_counts_with_phase(y, months_window, phase):
    if len(y) == 0:
        return np.array([], dtype=int)
    ys = y[phase:]
    B = len(ys) // months_window
    if B <= 0:
        return np.array([], dtype=int)
    trimmed = ys[:B * months_window]
    return trimmed.reshape(B, months_window).sum(axis=1)

def phase_offsets(months_window, K=5):
    K_eff = min(K, months_window)
    offsets = np.rint(np.linspace(0, months_window, num=K_eff+1)[:-1]).astype(int)
    return np.unique(offsets)

# -------------------------
# Global axis & overlaps
# -------------------------

def build_global_month_axis(node, keys):
    mmins, mmaxs = {}, {}
    for k in keys:
        df = node.dfs.get(k, None)
        if df is None or len(df) == 0:
            continue
        m = df["date"].dt.month + df["date"].dt.year * 12
        mmins[k] = int(m.min())
        mmaxs[k] = int(m.max())
    if not mmins:
        return {}, {}, None, None

    gmin = min(mmins.values()); gmax = max(mmaxs.values())
    L = gmax - gmin + 1

    y_by_src, start_on_global = {}, {}
    for k in keys:
        df = node.dfs.get(k, None)
        if df is None or len(df) == 0:
            continue
        m = df["date"].dt.month + df["date"].dt.year * 12
        m = m - gmin
        y_by_src[k] = np.bincount(m, minlength=L)
        start_on_global[k] = int((mmins[k] - gmin))
    return y_by_src, start_on_global, 0, L-1  # global axis [0..L-1]

def overlap_slice_for_sources(start_on_global, end_global, sources):
    starts = [start_on_global[s] for s in sources]
    start = max(starts)
    end = end_global
    if start > end:
        return None
    return (start, end)

def slice_counts(y, start, end):
    start = max(0, start); end = min(len(y)-1, end)
    if start > end:
        return np.array([], dtype=int)
    return y[start:end+1]

def expected_ok(table, min_expected=5):
    row_tot = table.sum(axis=1, keepdims=True)
    col_tot = table.sum(axis=0, keepdims=True)
    grand = table.sum()
    if grand == 0:
        return False
    expected = (row_tot @ col_tot) / grand
    return np.all(expected >= min_expected)

# -------------------------
# Poisson GLM (categorical bins) one-vs-rest via pairwise Šidák
# -------------------------

def poisson_catbins_one_vs_rest_p(y_by_src, start_on_global, end_global,
                                  all_sources, focal, months_window, phase,
                                  min_bins=2, min_expected=5):
    others = [s for s in all_sources if s != focal]
    pair_ps = []
    for other in others:
        ov = overlap_slice_for_sources(start_on_global, end_global, (focal, other))
        if not ov: 
            continue
        s, e = ov
        cA = bin_counts_with_phase(slice_counts(y_by_src[focal], s, e), months_window, phase)
        cB = bin_counts_with_phase(slice_counts(y_by_src[other], s, e), months_window, phase)
        Bn = min(len(cA), len(cB))
        if Bn < min_bins:
            continue
        table = np.column_stack([cA[:Bn], cB[:Bn]])
        if not expected_ok(table, min_expected):
            continue
        # Long DF (bin categorical)
        df_long = pd.DataFrame({
            "bin": np.repeat(np.arange(Bn), 2),
            "source": np.tile([focal, other], Bn),
            "count": table.flatten(order="F")  # by column
        })
        m_full = smf.glm("count ~ C(source) * C(bin)", data=df_long, family=sm.families.Poisson()).fit()
        m_red  = smf.glm("count ~ C(source) + C(bin)", data=df_long, family=sm.families.Poisson()).fit()
        lr = 2*(m_full.llf - m_red.llf)
        df_diff = m_full.df_model - m_red.df_model
        p = chi2.sf(lr, df_diff)
        if np.isfinite(p):
            pair_ps.append(p)
    if len(pair_ps) == 0:
        return np.nan
    p_min = np.min(pair_ps); K = len(pair_ps)
    return 1.0 - (1.0 - p_min)**K






# -------------------------
# Spline Poisson GLM (continuous time), one-vs-rest via pairwise Šidák
# -------------------------

def poisson_spline_one_vs_rest_p(node, all_sources, focal, df_s, min_months=12):
    y_by_src, start_on_global, _, end_global = build_global_month_axis(node, all_sources)
    if not y_by_src or focal not in y_by_src:
        return np.nan
    others = [s for s in all_sources if s != focal]
    pair_ps = []
    for other in others:
        if other not in y_by_src:
            continue
        ov = overlap_slice_for_sources(start_on_global, end_global, (focal, other))
        if not ov:
            continue
        s, e = ov
        if (e - s + 1) < min_months:
            continue
        t = np.arange(s, e+1)
        cA = slice_counts(y_by_src[focal], s, e)
        cB = slice_counts(y_by_src[other], s, e)
        df_long = pd.DataFrame({
            "time": np.concatenate([t, t]),
            "source": [focal]*len(t) + [other]*len(t),
            "count":  np.concatenate([cA, cB])
        })
        if df_long["count"].sum() < 20:
            continue
        try:
            m_full = smf.glm(f"count ~ C(source) * bs(time, df={df_s}, degree=3)",
                             data=df_long, family=sm.families.Poisson()).fit()
            m_red  = smf.glm(f"count ~ C(source) + bs(time, df={df_s}, degree=3)",
                             data=df_long, family=sm.families.Poisson()).fit()
        except Exception:
            continue
        lr = 2*(m_full.llf - m_red.llf)
        df_diff = m_full.df_model - m_red.df_model
        p = chi2.sf(lr, df_diff)
        if np.isfinite(p):
            pair_ps.append(p)
    if len(pair_ps)==0:
        return np.nan
    p_min = np.min(pair_ps); K = len(pair_ps)
    return 1.0 - (1.0 - p_min)**K





# -------------------------
# Main plotting (one line per source, p-only)
# -------------------------

def calc_uniqueness(node,
                                 keys=("arx_long","hf","dlw_long"),
                                 months_window_range=range(6, 25),
                                 phase_K=5,
                                 min_bins=2,
                                 min_expected=5,
                                 spline_df_grid=range(4, 11),
                                 min_overlap_months=12):

    y_by_src, start_on_global, _, end_global = build_global_month_axis(node, keys)
    sources = [k for k in keys if k in y_by_src]
    if len(sources) < 2:
        return

    node.temporality_uniqueness_scores = {}
    node.uniqueness_x = {}
    node.uniqueness_y = {}

    # ---------- Panels that depend on bin size (χ², categorical GLM, hybrid)
    for focal in sources:
        X_sizes = []
        Y_cat = []
        Y_cat_simes = []
        simes_p_list = []
        for W in months_window_range:
            phases = phase_offsets(W, K=phase_K)

            p_cat_phases = []

            for ph in phases:

                p_cat = poisson_catbins_one_vs_rest_p(
                    y_by_src, start_on_global, end_global, sources, focal,
                    months_window=W, phase=ph, min_bins=min_bins, min_expected=min_expected
                )
                if np.isfinite(p_cat):
                    p_cat_phases.append(p_cat)


            # Phase combine for each method
            any_valid = False
            if len(p_cat_phases):
                pmin = np.min(p_cat_phases); Kp = len(p_cat_phases)
                p_sidak = 1.0 - (1.0 - pmin)**Kp
                p_simes = simes_combine_pvalues(p_cat_phases)
                simes_p_list.append(p_simes)
                # keep same X by appending at same W
                if W not in X_sizes:
                    X_sizes.append(W)
                Y_cat.append(-np.log10(max(p_sidak, 1e-300))); any_valid = True
                Y_cat_simes.append(-np.log10(max(p_simes, 1e-300))); any_valid = True

        # plot lines for this focal source
        if len(X_sizes) and len(Y_cat):
            #ax_cat.plot(X_sizes[:len(Y_cat)], Y_cat, label=f"{focal}")
            node.uniqueness_x[focal] = X_sizes[:len(Y_cat_simes)]
            node.uniqueness_y[focal] = Y_cat_simes
    

        node.temporality_uniqueness_scores[focal] = simes_combine_pvalues(simes_p_list)


    
    #if not(len(X_sizes) and len(Y_cat)):
    #    return
    #else:
    if 1:
        for child in node.children:
            calc_uniqueness(child,
                     keys,
                     months_window_range,
                     phase_K,
                     min_bins,
                     min_expected,
                     spline_df_grid,
                     min_overlap_months)
