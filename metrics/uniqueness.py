import numpy as np
import pandas as pd

from metrics.nonuniformity import simes_combine_pvalues


def month_id_from_ts(ts):
    ts = pd.to_datetime(ts)
    return int(ts.year * 12 + ts.month)

def compute_constant_overlap_from_root(root, sources, date_col="date"):
    """
    Computes the intersection overlap period across all sources using the ROOT node,
    assumed to contain all occurrences for each source.

    Returns:
      gmin, gmax: integer month ids (year*12 + month) defining inclusive overlap window
    """
    mins = []
    maxs = []
    for s in sources:
        df = root.dfs.get(s, None)
        if df is None or len(df) == 0:
            raise ValueError(f"Root has no data for source '{s}'")

        t = pd.to_datetime(df[date_col])
        mins.append(t.min())
        maxs.append(t.max())

    ov_start = max(mins)   # latest start
    ov_end   = min(maxs)   # earliest end
    if ov_start > ov_end:
        raise ValueError(f"No common overlap across sources: {ov_start} > {ov_end}")

    gmin = month_id_from_ts(ov_start)
    gmax = month_id_from_ts(ov_end)
    return gmin, gmax


def build_month_axis(node, sources, gmin, gmax, date_col="date"):
    """
    Builds per-source monthly count arrays restricted to the FIXED overlap [gmin..gmax]
    (inclusive). Months inside overlap with no occurrences become 0.
    """
    L = (gmax - gmin) + 1
    y_by_src = {}

    for s in sources:
        df = node.dfs.get(s, None)
        if df is None or len(df) == 0:
            y_by_src[s] = np.zeros(L, dtype=int)
            continue

        m = df[date_col].dt.year * 12 + df[date_col].dt.month
        m = m[(m >= gmin) & (m <= gmax)]
        if len(m) == 0:
            y_by_src[s] = np.zeros(L, dtype=int)
            continue

        idx = (m - gmin).astype(int).to_numpy()
        y_by_src[s] = np.bincount(idx, minlength=L).astype(int)

    return y_by_src


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

def poisson_spline_focal_vs_rest_p(
    y_by_src,
    sources,
    focal,
    df_s,
    min_months=12,
    min_total_count=20,
):
    """

    Reduced: count ~ C(source) + bs(time, df=df_s, degree=3)
    Full:    count ~ C(source) * bs(time, df=df_s, degree=3)
    """
    if focal not in y_by_src:
        return np.nan

    others = [s for s in sources if s != focal and s in y_by_src]
    if not others:
        return np.nan

    y_focal = y_by_src[focal].astype(float)
    y_rest = np.zeros_like(y_focal)
    for s in others:
        y_rest += y_by_src[s].astype(float)

    T = len(y_focal)
    if T < min_months:
        return np.nan
    if (y_focal.sum() + y_rest.sum()) < min_total_count:
        return np.nan

    t = np.arange(T, dtype=float)
    df_long = pd.DataFrame({
        "time": np.concatenate([t, t]),
        "source": (["focal"] * T) + (["rest"] * T),
        "count": np.concatenate([y_focal, y_rest]),
    })

    try:
        m_red = smf.glm(
            f"count ~ C(source) + bs(time, df={df_s}, degree=3)",
            data=df_long, family=sm.families.Poisson()
        ).fit()
        m_full = smf.glm(
            f"count ~ C(source) * bs(time, df={df_s}, degree=3)",
            data=df_long, family=sm.families.Poisson()
        ).fit()
    except Exception:
        return np.nan

    lr = 2.0 * (m_full.llf - m_red.llf)
    df_diff = m_full.df_model - m_red.df_model
    if df_diff <= 0:
        return np.nan

    p = chi2.sf(lr, df_diff)
    if np.isfinite(p):
        return p
    else:
        return np.nan


def calc_uniqueness(
    node,
    keys=("arx_long", "hf", "dlw_long"),
    spline_df_grid=range(4, 11),
    min_overlap_months=12,
    min_total_count=20,
    _root=None,
    _gmin=None,
    _gmax=None,
):
    # Identify root on first entry
    if _root is None:
        _root = node

    # Compute constant overlap
    if _gmin is None or _gmax is None:
        _gmin, _gmax = compute_constant_overlap_from_root(_root, keys, date_col="date")

    # Build monthly axis for this node
    y_by_src = build_month_axis(
        node, keys, _gmin, _gmax, date_col="date"
    )
    sources = [k for k in keys if k in y_by_src]  # here it will always be keys
    if len(sources) < 2:
        return

    node.temporality_uniqueness_scores = {}
    node.uniqueness_x = {}
    node.uniqueness_y = {}

    # --- One-vs-rest pooled spline metric
    for focal in sources:
        X_df, Y_score, p_list = [], [], []

        for df_s in spline_df_grid:
            p = poisson_spline_focal_vs_rest_p(
                y_by_src=y_by_src,
                sources=sources,
                focal=focal,
                df_s=df_s,
                min_months=min_overlap_months,
                min_total_count=min_total_count,
            )

            if np.isfinite(p):
                X_df.append(df_s)
                Y_score.append(-np.log10(max(p, 1e-300)))
                p_list.append(p)

        node.uniqueness_x[focal] = X_df
        node.uniqueness_y[focal] = Y_score
        node.temporality_uniqueness_scores[focal] = simes_combine_pvalues(p_list) if p_list else np.nan

    # recurse with same gmin/gmax
    for child in node.children:
        calc_uniqueness(
            child,
            keys=keys,
            spline_df_grid=spline_df_grid,
            min_overlap_months=min_overlap_months,
            min_total_count=min_total_count,
            _root=_root,
            _gmin=_gmin,
            _gmax=_gmax,
        )
