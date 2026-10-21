import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --- 設定 ---
LOG_DIR = Path("./logs")
HEATMAP_PAGE_SIZE = 20

st.set_page_config(
    page_title="FL Simulation Dashboard",
    page_icon="📊",
    layout="wide",
)


def load_experiments() -> list[str]:
    """logsディレクトリ内の実験フォルダ一覧を取得（新しい順）"""
    if not LOG_DIR.exists():
        return []
    return sorted((d.name for d in LOG_DIR.iterdir() if d.is_dir()), reverse=True)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


@st.cache_data
def load_data(exp_id: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """指定された実験IDの設定・指標・クライアント割り当てを読み込む"""
    exp_path = LOG_DIR / exp_id

    config_path = exp_path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    log_df = _read_csv(exp_path / "log.csv")
    alloc_df = _read_csv(exp_path / "client_allocation.csv")
    return config, log_df, alloc_df


def render_kpis(log_df: pd.DataFrame):
    latest = log_df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Round", int(latest["Round"]))
    col2.metric("Accuracy", f"{latest['Accuracy']:.2f}%")
    col3.metric("Train Loss", f"{latest['Train Loss']:.4f}")
    col4.metric("Val Loss", f"{latest['Val Loss']:.4f}")


def render_training_metrics(log_df: pd.DataFrame):
    st.subheader("📈 Training Metrics")
    tab1, tab2 = st.tabs(["Accuracy & Loss", "Time per Round"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig_acc = px.line(
                log_df, x="Round", y="Accuracy", title="Global Accuracy", markers=True
            )
            st.plotly_chart(fig_acc, width="stretch")
        with c2:
            loss_df = log_df.melt(
                id_vars=["Round"],
                value_vars=["Train Loss", "Val Loss"],
                var_name="Type",
                value_name="Loss",
            )
            fig_loss = px.line(
                loss_df,
                x="Round",
                y="Loss",
                color="Type",
                title="Loss Evolution",
                markers=True,
            )
            st.plotly_chart(fig_loss, width="stretch")

    with tab2:
        if "Time" in log_df.columns:
            fig_time = px.bar(
                log_df, x="Round", y="Time", title="Execution Time per Round (seconds)"
            )
            st.plotly_chart(fig_time, width="stretch")
        else:
            st.info("Time data not available.")


def render_cluster_sizes(log_df: pd.DataFrame):
    st.subheader("👥 Cluster Sizes")
    center_cols = [c for c in log_df.columns if c.startswith("Center_")]
    if not center_cols:
        return

    cluster_df = log_df.melt(
        id_vars=["Round"],
        value_vars=center_cols,
        var_name="Center",
        value_name="Count",
    )
    cluster_df["Center"] = cluster_df["Center"].str.replace(
        r"Center_(\d+)_Clients", r"C\1", regex=True
    )
    fig_cluster = px.area(
        cluster_df,
        x="Round",
        y="Count",
        color="Center",
        title="Client Distribution per Center",
    )
    st.plotly_chart(fig_cluster, width="stretch")


def render_allocation_heatmap(alloc_df: pd.DataFrame, num_centers: int):
    st.subheader("🗺️ Client Allocation Heatmap")
    if alloc_df.empty:
        st.info("No allocation data found (client_allocation.csv).")
        return

    st.markdown("""
    **見方**: 縦軸が各ユーザー、横軸がラウンドの進行を表します。
    ユーザーは「最終ラウンドで所属しているセンター順」に自動ソートされています。
    色が安定して帯状になっている部分は、所属が固定化していることを示します。
    空白はそのラウンドに参加しなかったことを示します。
    """)

    # 行: ユーザー、列: ラウンド。最終ラウンドの所属センター順にソート
    heatmap_data = alloc_df.set_index("Round").T
    heatmap_data = heatmap_data.sort_values(by=heatmap_data.columns[-1])

    # ページネーション（動作を軽くするため分割表示）
    total_writers = len(heatmap_data)
    if total_writers > HEATMAP_PAGE_SIZE:
        st.caption(
            f"Total Writers: {total_writers}. 表示範囲を選択してください（動作を軽くするため分割表示します）。"
        )
        start_idx = st.slider(
            "Display Range (Start Index)",
            min_value=0,
            max_value=total_writers - HEATMAP_PAGE_SIZE,
            value=0,
            step=HEATMAP_PAGE_SIZE,
        )
        heatmap_data = heatmap_data.iloc[start_idx : start_idx + HEATMAP_PAGE_SIZE * 2]

    fig_heat = px.imshow(
        heatmap_data,
        labels={"x": "Round", "y": "Writer ID", "color": "Center ID"},
        aspect="auto",
        color_continuous_scale="Viridis",
        origin="lower",
    )
    fig_heat.update_traces(xgap=1, ygap=1)
    fig_heat.update_layout(
        height=600,
        xaxis_title="Round",
        yaxis_title="Writer ID (Sorted by Final Center)",
        coloraxis_colorbar={
            "title": "Center ID",
            "tickvals": list(range(num_centers)),
            "ticktext": [f"C{i}" for i in range(num_centers)],
        },
    )
    st.plotly_chart(fig_heat, width="stretch")


def main():
    st.title("📊 Federated Learning Visualization")

    # --- サイドバー: 実験選択 ---
    st.sidebar.header("Experiment Selection")
    experiments = load_experiments()
    if not experiments:
        st.warning(f"No experiments found in {LOG_DIR}")
        return

    selected_exp = st.sidebar.selectbox("Select Experiment ID", experiments)
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()  # キャッシュクリアして再読み込み

    config, log_df, alloc_df = load_data(selected_exp)

    with st.expander("📝 Experiment Configuration", expanded=False):
        st.json(config)

    if log_df.empty:
        st.error("Log data (log.csv) is empty or not found.")
        return

    render_kpis(log_df)
    st.divider()
    render_training_metrics(log_df)
    render_cluster_sizes(log_df)
    render_allocation_heatmap(alloc_df, config.get("num_centers", 5))


if __name__ == "__main__":
    main()
