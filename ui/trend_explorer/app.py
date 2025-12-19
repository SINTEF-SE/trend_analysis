"""Standalone Streamlit UI for exploring AI taxonomy trends."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TREND_DATA_DIR = REPO_ROOT / "output_labels"
TAXONOMY_PATH = REPO_ROOT / "taxonomies" / "ai_taxonomies.yaml"
TopicPath = Tuple[str, ...]

BRANCH_DEFINITIONS: Tuple[Tuple[str, str], ...] = (
    ("Model architecture", "architecture"),
    ("AI problem type", "problem"),
    ("Learning paradigm", "paradigm"),
    ("Application domain", "application"),
)
BRANCH_KEY_MAP: Dict[str, str] = {branch: key for branch, key in BRANCH_DEFINITIONS}


def load_ai_taxonomy(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find AI taxonomy at {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    # Accept either underscore or space spelling at root.
    taxonomy = data.get("AI_Taxonomy") or data.get("AI Taxonomy") or data
    if not isinstance(taxonomy, dict):
        raise ValueError("Loaded taxonomy has unexpected structure.")
    return taxonomy


def remove_not_relevant(node: Dict | List | None):
    """Recursively drop any 'Not relevant' entries from the taxonomy tree for navigation."""

    if isinstance(node, dict):
        return {k: remove_not_relevant(v) for k, v in node.items() if k != "Not relevant"}
    if isinstance(node, list):
        return [item for item in node if item != "Not relevant"]
    return node


AI_TAXONOMY = remove_not_relevant(load_ai_taxonomy(TAXONOMY_PATH))
ROOT_NODE = "AI Taxonomy"
TAXONOMY_TREE: Dict[str, Dict] = {ROOT_NODE: AI_TAXONOMY}
TOP_LEVEL_BRANCHES: Tuple[str, ...] = tuple(AI_TAXONOMY.keys())
ARXIV_DATA_PATH = TREND_DATA_DIR / "arx_long_trends.csv.bz2"
DLW_DATA_PATH = TREND_DATA_DIR / "dlw_long_trends.csv.bz2"
HF_DATA_PATH = TREND_DATA_DIR / "hf_trends.csv.bz2"


@dataclass(frozen=True)
class TopicSummary:
    path: TopicPath
    paper_count: int

    @property
    def label(self) -> str:
        return " / ".join(self.path)


@dataclass(frozen=True)
class DataSourceConfig:
    key: str
    label: str
    path: Path
    id_column: str
    primary_date_column: str
    parse_dates: Tuple[str, ...]
    entity_label: str
    rename_columns: Dict[str, str] = None
    secondary_date_column: Optional[str] = None
    link_template: Optional[str] = None
    link_label: Optional[str] = None
    link_text: str = "Open"

    def resolved_rename_map(self) -> Dict[str, str]:
        return self.rename_columns or {}


@dataclass
class SourceView:
    key: str
    config: DataSourceConfig
    full_df: pd.DataFrame
    filtered_df: pd.DataFrame
    branch_column: str
    overall_counts: Dict[TopicPath, int]
    filtered_counts: Dict[TopicPath, int]
    monthly_counts: pd.DataFrame


DATA_SOURCES: Dict[str, DataSourceConfig] = {
    "arxiv": DataSourceConfig(
        key="arxiv",
        label="arXiv",
        path=ARXIV_DATA_PATH,
        id_column="id",
        primary_date_column="date",
        parse_dates=("date",),
        entity_label="Papers",
        link_template="https://arxiv.org/abs/{id}",
        link_label="arXiv",
    ),
    "hf": DataSourceConfig(
        key="hf",
        label="Hugging Face",
        path=HF_DATA_PATH,
        id_column="modelId",
        primary_date_column="date",
        parse_dates=("date", "last_modified"),
        entity_label="Models",
        secondary_date_column="last_modified",
        link_template="https://huggingface.co/{modelId}",
        link_label="Hub",
    ),
    "dlw": DataSourceConfig(
        key="dlw",
        label="DL Weekly",
        path=DLW_DATA_PATH,
        id_column="record_id",
        primary_date_column="date",
        parse_dates=("date",),
        entity_label="Entries",
        rename_columns={"Unnamed: 0": "record_id", "": "record_id"},
    ),
}

EXTRA_TABLE_COLUMNS: Dict[str, List[str]] = {
    "arxiv": ["categories"],
    "hf": [],
    "dlw": [],
}


def _iter_paths(tree: Dict | List | str | None, prefix: TopicPath = ()) -> Iterable[TopicPath]:
    if isinstance(tree, dict):
        for key, child in tree.items():
            new_prefix = prefix + (key,)
            yield new_prefix
            if child is None:
                continue
            yield from _iter_paths(child, new_prefix)
    elif isinstance(tree, list):
        for item in tree:
            new_prefix = prefix + (item,)
            yield new_prefix
    elif tree is None:
        return
    else:
        raise TypeError(f"Unsupported tree node type: {type(tree)}")


def _get_subtree(tree: Dict | List | None, path: TopicPath) -> Dict | List | None:
    subtree = tree
    for key in path:
        if isinstance(subtree, dict):
            if key not in subtree:
                raise KeyError(f"Node '{key}' not found under path {' / '.join(path[:-1])}")
            subtree = subtree[key]
        elif isinstance(subtree, list):
            if key not in subtree:
                raise KeyError(f"Leaf '{key}' not found under path {' / '.join(path[:-1])}")
            subtree = None
        elif subtree is None:
            raise KeyError(f"Cannot descend beyond leaf at {' / '.join(path)}")
        else:
            raise KeyError(f"Cannot descend into node at {' / '.join(path)}")
    return subtree


def list_all_topic_paths() -> List[TopicPath]:
    return [path for path in _iter_paths(TAXONOMY_TREE)]


def branch_path_column(branch: str) -> str:
    if branch not in BRANCH_KEY_MAP:
        raise KeyError(f"Unknown branch '{branch}'")
    return f"{BRANCH_KEY_MAP[branch]}_path"


def determine_branch(selected_path: TopicPath) -> str:
    if len(selected_path) >= 2 and selected_path[1] in BRANCH_KEY_MAP:
        return selected_path[1]
    return BRANCH_DEFINITIONS[0][0]


def normalise_selected_path(selected_path: TopicPath) -> TopicPath:
    if not selected_path:
        default_branch = BRANCH_DEFINITIONS[0][0]
        return (ROOT_NODE, default_branch)
    if selected_path[0] != ROOT_NODE:
        return (ROOT_NODE,) + selected_path
    if len(selected_path) == 1:
        default_branch = BRANCH_DEFINITIONS[0][0]
        return (ROOT_NODE, default_branch)
    return selected_path


def normalise_predicted_tag(raw_tags: object) -> Tuple[str, ...]:
    if isinstance(raw_tags, (list, tuple)):
        return tuple(raw_tags)
    if isinstance(raw_tags, str):
        try:
            parsed = ast.literal_eval(raw_tags)
        except (ValueError, SyntaxError):
            return tuple()
        if isinstance(parsed, (list, tuple)):
            return tuple(parsed)
    return tuple()


def segment_branch_paths(tags: Tuple[str, ...]) -> Dict[str, TopicPath]:
    branch_paths: Dict[str, TopicPath] = {}
    current_branch: Optional[str] = None
    current_path: List[str] = []

    for label in tags:
        if label == "Not relevant":
            # Drop "Not relevant" and keep path at parent level.
            continue
        if label in BRANCH_KEY_MAP:
            if current_branch is not None and current_path:
                branch_paths[current_branch] = tuple(current_path)
            current_branch = label
            current_path = [ROOT_NODE, label]
        else:
            if current_branch is None:
                continue
            current_path.append(label)

    if current_branch is not None and current_path:
        branch_paths[current_branch] = tuple(current_path)

    for branch in BRANCH_KEY_MAP:
        branch_paths.setdefault(branch, (ROOT_NODE, branch))

    # Collapse any lingering "Not relevant" labels to their parent.
    for branch, path in branch_paths.items():
        branch_paths[branch] = collapse_not_relevant_path(path)

    return branch_paths


def filter_by_branch_path(df: pd.DataFrame, branch: str, target_path: TopicPath) -> pd.DataFrame:
    adjusted_path = normalise_selected_path(target_path)
    column_name = branch_path_column(branch)
    prefix_len = len(adjusted_path)
    if prefix_len <= 2:
        return df.copy()
    mask = df[column_name].apply(lambda topic: topic[:prefix_len] == adjusted_path)
    return df[mask].copy()


def prepare_source_views(selected_path: TopicPath, branch: str, sources: Dict[str, pd.DataFrame]) -> Dict[str, SourceView]:
    adjusted_path = normalise_selected_path(selected_path)
    views: Dict[str, SourceView] = {}
    column_name = branch_path_column(branch)

    for key, df in sources.items():
        config = DATA_SOURCES[key]
        branch_paths = tuple(df[column_name])
        overall_counts = compute_topic_counts(branch_paths)

        filtered_df = filter_by_branch_path(df, branch, adjusted_path)
        if filtered_df.empty:
            filtered_counts: Dict[TopicPath, int] = {}
            monthly_counts = pd.DataFrame(columns=["month", "count"])
        else:
            filtered_counts = compute_topic_counts(tuple(filtered_df[column_name]))
            monthly_counts = compute_monthly_counts(filtered_df, config.primary_date_column, config.id_column)

        views[key] = SourceView(
            key=key,
            config=config,
            full_df=df,
            filtered_df=filtered_df,
            branch_column=column_name,
            overall_counts=overall_counts,
            filtered_counts=filtered_counts,
            monthly_counts=monthly_counts,
        )

    return views


def build_display_table(view: SourceView, branch: str) -> Tuple[pd.DataFrame, Dict[str, st.column_config.Column]]:
    df = view.filtered_df.copy()
    if df.empty:
        return df, {}

    config = view.config
    display_columns: List[str] = [config.id_column, config.primary_date_column]

    if config.secondary_date_column and config.secondary_date_column in df.columns:
        display_columns.append(config.secondary_date_column)

    for column in EXTRA_TABLE_COLUMNS.get(view.key, []):
        if column in df.columns:
            display_columns.append(column)

    path_display_column = f"{BRANCH_KEY_MAP[branch]}_path_display"
    df[path_display_column] = df[view.branch_column].apply(format_branch_path)
    display_columns.append(path_display_column)

    column_config: Dict[str, st.column_config.Column] = {
        path_display_column: st.column_config.TextColumn("Taxonomy path"),
    }

    if config.link_template:
        link_column = f"{view.key}_link"
        df[link_column] = df[config.id_column].apply(
            lambda value: config.link_template.format(**{config.id_column: value})
            if pd.notna(value)
            else ""
        )
        display_columns.append(link_column)
        column_config[link_column] = st.column_config.LinkColumn(
            config.link_label or config.label,
            display_text=config.link_text,
        )

    if pd.api.types.is_datetime64_any_dtype(df[config.primary_date_column]):
        df[config.primary_date_column] = df[config.primary_date_column].dt.date

    if config.secondary_date_column and config.secondary_date_column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[config.secondary_date_column]):
            df[config.secondary_date_column] = df[config.secondary_date_column].dt.date

    df = df[display_columns]
    df.sort_values(config.primary_date_column, ascending=False, inplace=True)

    readable_names = {
        config.id_column: config.id_column.replace("_", " ").title(),
        config.primary_date_column: config.primary_date_column.replace("_", " ").title(),
    }
    if config.secondary_date_column and config.secondary_date_column in df.columns:
        readable_names[config.secondary_date_column] = config.secondary_date_column.replace("_", " ").title()
    if config.key in EXTRA_TABLE_COLUMNS:
        for column in EXTRA_TABLE_COLUMNS[config.key]:
            if column in df.columns:
                readable_names[column] = column.replace("_", " ").title()
    readable_names[path_display_column] = "Taxonomy path"

    df.rename(columns=readable_names, inplace=True)

    return df, column_config


def render_source_view_tab(view: SourceView, branch: str, selected_path: TopicPath) -> None:
    config = view.config
    entity_label = config.entity_label
    filtered_df = view.filtered_df

    st.metric(entity_label, len(filtered_df))

    if filtered_df.empty:
        st.info(f"No {entity_label.lower()} mapped to this topic in {config.label}.")
        render_sunburst(view.overall_counts, highlight_path=selected_path, dataset_label=config.label)
        return

    primary_dates = filtered_df[config.primary_date_column]
    first_date = primary_dates.min()
    last_date = primary_dates.max()
    if pd.notna(first_date) and pd.notna(last_date):
        st.caption(f"Time span: {first_date.date()} → {last_date.date()}")

    if config.secondary_date_column and config.secondary_date_column in filtered_df.columns:
        secondary_dates = filtered_df[config.secondary_date_column].dropna()
        if not secondary_dates.empty:
            latest_secondary = secondary_dates.max()
            label = config.secondary_date_column.replace("_", " ").title()
            st.caption(f"Latest {label}: {latest_secondary.date()}")

    if view.monthly_counts.empty:
        st.caption("No monthly activity to display.")
    else:
        st.write("Monthly activity")
        trend_fig = px.line(
            view.monthly_counts,
            x="month",
            y="count",
            markers=True,
            labels={"month": "Month", "count": f"{entity_label} per month"},
        )
        trend_fig.update_layout(hovermode="x unified")
        trend_fig.update_xaxes(dtick="M12", tickformat="%Y", tickangle=0)
        st.plotly_chart(trend_fig, use_container_width=True)

    child_summaries = child_topic_summaries(selected_path, view.filtered_counts)
    if child_summaries:
        child_df = pd.DataFrame(
            {
                "Topic": [format_branch_path(summary.path) for summary in child_summaries],
                entity_label: [summary.paper_count for summary in child_summaries],
            }
        )
        st.write(f"Immediate subtopics ({entity_label.lower()})")
        st.table(child_df)

    render_sunburst(view.overall_counts, highlight_path=selected_path, dataset_label=config.label)

    display_df, column_config = build_display_table(view, branch)
    if not display_df.empty:
        st.write("Matching items")
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config=column_config,
        )


def _load_single_source(config: DataSourceConfig) -> pd.DataFrame:
    if not config.path.exists():
        raise FileNotFoundError(f"Cannot find trend data for {config.label} at {config.path}")

    parse_dates = list(config.parse_dates)
    df = pd.read_csv(config.path, compression="bz2", parse_dates=parse_dates)
    rename_map = config.resolved_rename_map()
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    df["predicted_tag"] = df["predicted_tag"].apply(normalise_predicted_tag)
    branch_mappings = df["predicted_tag"].apply(segment_branch_paths)

    for branch, key in BRANCH_DEFINITIONS:
        column_name = f"{key}_path"
        df[column_name] = branch_mappings.apply(lambda mapping, branch_name=branch: mapping[branch_name])

    df["source_key"] = config.key
    df["source_label"] = config.label
    df["entity_label"] = config.entity_label
    df["link_template"] = config.link_template
    df["link_label"] = config.link_label or config.label
    df["link_text"] = config.link_text

    return df


@st.cache_data(show_spinner=False)
def load_all_sources() -> Dict[str, pd.DataFrame]:
    return {key: _load_single_source(config) for key, config in DATA_SOURCES.items()}


@st.cache_data(show_spinner=False)
def compute_topic_counts(paths: Iterable[TopicPath]) -> Dict[TopicPath, int]:
    counts: Dict[TopicPath, int] = defaultdict(int)
    for path in tuple(paths):
        if not path:
            continue
        for depth in range(1, len(path) + 1):
            counts[path[:depth]] += 1
    return dict(counts)


def compute_monthly_counts(df: pd.DataFrame, date_column: str, id_column: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "count"])

    monthly_counts = (
        df.set_index(date_column)[id_column]
        .resample("MS")
        .count()
        .rename("count")
        .reset_index()
        .rename(columns={date_column: "month"})
    )
    return monthly_counts


def calculate_curve(dates: Iterable[pd.Timestamp], search_radius: Tuple[int, str] = (25, "W"), n_steps: int = 100) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Smooth publication dates with a mirrored triangular kernel."""

    dates_series = pd.Series(dates).dropna()
    dates = pd.to_datetime(dates_series).sort_values().to_list()
    if not dates:
        return pd.DatetimeIndex([], name="timestamp"), np.array([])

    min_date = dates[0]
    max_date = dates[-1]
    X = pd.date_range(start=min_date, end=max_date, periods=n_steps)

    radius_td = pd.Timedelta(*search_radius)
    sr_ns = np.timedelta64(radius_td.value, "ns")

    D = np.array(dates, dtype="datetime64[ns]")

    # mirror at month start instead of as the first/last doc, to avoid unfairly high frequency at the edge
    min_date = min_date.to_period("M").to_timestamp(how="start")
    max_date = max_date.to_period("M").to_timestamp(how="end")

    min_np = np.array(min_date, dtype="datetime64[ns]")
    max_np = np.array(max_date, dtype="datetime64[ns]")
    left_mask = (D - min_np) < sr_ns
    right_mask = (max_np - D) < sr_ns

    D_left = (min_np + (min_np - D[left_mask]))
    D_right = (max_np + (max_np - D[right_mask]))
    D_all = np.concatenate([D, D_left, D_right]) if (D_left.size or D_right.size) else D

    X_np = X.to_numpy(dtype="datetime64[ns]")
    dist = np.abs(X_np[:, None] - D_all[None, :])
    weights = (sr_ns - dist) / sr_ns
    weights = np.where(dist <= sr_ns, weights.astype(float), 0.0)
    Y = weights.sum(axis=1)

    # standardize to documents per month
    Y = Y * (pd.Timedelta(30.44, "D") / radius_td) # 30.44 days per month avg

    return X, Y


def build_smoothed_trend_dataframe(views: Dict[str, SourceView], search_radius_weeks: int, n_steps: int = 150) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    radius = (search_radius_weeks, "W")

    for view in views.values():
        filtered = view.filtered_df
        if filtered.empty:
            continue

        primary_dates = filtered[view.config.primary_date_column].dropna()
        if primary_dates.empty:
            continue

        X, Y = calculate_curve(primary_dates, search_radius=radius, n_steps=n_steps)
        if X.empty or Y.size == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": X,
                    "value": Y,
                    "dataset": view.config.label,
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=["timestamp", "value", "dataset"])

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    return combined


def child_topic_summaries(path: TopicPath, counts: Dict[TopicPath, int]) -> List[TopicSummary]:
    try:
        subtree = _get_subtree(TAXONOMY_TREE, path)
    except KeyError:
        return []

    if isinstance(subtree, dict):
        child_names = subtree.keys()
    elif isinstance(subtree, list):
        child_names = subtree
    elif subtree is None:
        return []
    else:
        return []

    summaries = []
    for child in child_names:
        child_path = path + (child,)
        summaries.append(TopicSummary(child_path, counts.get(child_path, 0)))
    summaries.sort(key=lambda summary: summary.paper_count, reverse=True)
    return summaries


@lru_cache(maxsize=1)
def all_topic_paths() -> List[TopicPath]:
    return list_all_topic_paths()


def build_hierarchy_dataframe(counts: Dict[TopicPath, int]) -> pd.DataFrame:
    if not counts:
        return pd.DataFrame()

    rows = []
    for path, value in counts.items():
        node_id = " / ".join(path)
        parent_id = " / ".join(path[:-1]) if len(path) > 1 else ""
        label = path[-1]
        rows.append({"id": node_id, "parent": parent_id, "label": label, "value": value})
    return pd.DataFrame(rows)


def format_path(path: TopicPath) -> str:
    return " / ".join(path)


def format_branch_path(path: TopicPath) -> str:
    if not path:
        return ""
    if path[0] == ROOT_NODE:
        return " / ".join(path[1:])
    return " / ".join(path)


def collapse_not_relevant_path(path: TopicPath) -> TopicPath:
    """Remove any 'Not relevant' label by truncating to its parent."""

    if "Not relevant" not in path:
        return path

    idx = path.index("Not relevant")
    # Always keep at least root + branch to preserve grouping
    return path[: max(2, idx)]


def child_options(node: Dict | List | None) -> List[str]:
    """Return available child labels, skipping 'Not relevant' and injecting 'Other' when useful."""

    if isinstance(node, dict):
        children = [child for child in node.keys() if child != "Not relevant"]
    elif isinstance(node, list):
        children = [child for child in node if child != "Not relevant"]
    else:
        children = []

    if children and "Other" not in children:
        children.append("Other")

    return sorted(children)


def topic_selector() -> TopicPath:
    st.sidebar.subheader("Taxonomy Navigator")

    branch_options = [branch for branch, _ in BRANCH_DEFINITIONS]
    branch = st.sidebar.radio("Axis", branch_options, index=0, key="branch_selector")

    path: List[str] = [ROOT_NODE, branch]
    current_node = AI_TAXONOMY.get(branch, {})
    depth = 0

    while True:
        label = f"Level {depth + 1}"
        if isinstance(current_node, dict):
            options: List[Optional[str]] = [None] + child_options(current_node)
            selection = st.sidebar.selectbox(
                label,
                options,
                format_func=lambda option: f"Stay at {path[-1]}" if option is None else option,
                key=f"{branch}_selector_{depth}_{'_'.join(path).replace(' ', '_')}"
            )
            if selection is None:
                break
            path.append(selection)
            current_node = current_node.get(selection) if selection != "Other" else None
            depth += 1
        elif isinstance(current_node, list):
            leaf_options: List[Optional[str]] = [None] + child_options(current_node)
            selection = st.sidebar.selectbox(
                label,
                leaf_options,
                format_func=lambda option: f"Stay at {path[-1]}" if option is None else option,
                key=f"{branch}_selector_leaf_{depth}_{'_'.join(path).replace(' ', '_')}"
            )
            if selection is None:
                break
            path.append(selection)
            break
        elif current_node is None:
            break
        else:
            break

    return tuple(path)


def render_sunburst(
    counts: Dict[TopicPath, int],
    highlight_path: Optional[TopicPath] = None,
    dataset_label: str = "arXiv",
) -> None:
    hierarchy_df = build_hierarchy_dataframe(counts)
    if hierarchy_df.empty:
        st.warning(f"Hierarchy overview unavailable for {dataset_label}. No topic counts computed.")
        return

    fig = px.sunburst(
        hierarchy_df,
        names="label",
        parents="parent",
        values="value",
        ids="id",
        branchvalues="total"
    )

    if highlight_path:
        highlight_id = " / ".join(highlight_path)
        ids = list(fig.data[0].ids)
        marker = fig.data[0].marker
        base_colors = list(marker.colors) if marker.colors is not None else [None] * len(ids)
        if highlight_id in ids:
            updated_colors = base_colors.copy()
            for idx, node_id in enumerate(ids):
                if base_colors[idx] is None:
                    updated_colors[idx] = px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)]
            for depth in range(1, len(highlight_path) + 1):
                ancestor_id = " / ".join(highlight_path[:depth])
                if ancestor_id in ids:
                    ancestor_idx = ids.index(ancestor_id)
                    updated_colors[ancestor_idx] = "#1f77b4"
            marker.colors = updated_colors
            marker.line.width = 1
            marker.line.color = "white"

    fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
    st.subheader(f"Hierarchy overview ({dataset_label})")
    st.plotly_chart(fig, config={"responsive": True})


def render_branch_overview(selected_path: TopicPath, views: Dict[str, SourceView], branch: str) -> None:
    st.header(format_branch_path(selected_path))

    smoothing_weeks = st.slider(
        "Smoothing window (weeks)",
        min_value=4,
        max_value=100,
        value=25,
        help="Controls the triangular kernel radius used to smooth the activity curves.",
        key="smoothing_window_weeks",
    )

    smoothed_df = build_smoothed_trend_dataframe(views, smoothing_weeks)

    if not smoothed_df.empty:
        st.subheader("Smoothed activity comparison")
        curve_fig = px.line(
            smoothed_df,
            x="timestamp",
            y="value",
            color="dataset",
            category_orders={"dataset": ["arXiv", "Hugging Face", "DL Weekly"]},
            color_discrete_map={
                "arXiv": "#1f77b4",
                "Hugging Face": "#ff7f0e",
                "DL Weekly": "#949827",
                },
            labels={"timestamp": "Date", "value": "Monthly activity", "dataset": "Source"},

        )
        curve_fig.update_layout(hovermode="x unified")
        curve_fig.update_traces(mode="lines+markers")
        curve_fig.update_xaxes(dtick="M12", tickformat="%Y", tickangle=0)
        st.plotly_chart(curve_fig, use_container_width=True)
    else:
        st.info("No activity found for this topic across the available sources.")

    tabs = st.tabs([view.config.label for view in views.values()])
    for tab, view in zip(tabs, views.values()):
        with tab:
            render_source_view_tab(view, branch, selected_path)


def main() -> None:
    st.set_page_config(page_title="AI Taxonomy Trend Explorer", layout="wide")
    st.title("AI Taxonomy Trend Explorer")
    st.caption(
        "Explore AI taxonomy topics across the long-range arXiv and DL Weekly datasets plus Hugging Face models, with adjustable smoothing."
    )

    sources = load_all_sources()
    selected_path = normalise_selected_path(topic_selector())
    branch = determine_branch(selected_path)
    views = prepare_source_views(selected_path, branch, sources)

    render_branch_overview(selected_path, views, branch)


if __name__ == "__main__":
    main()
