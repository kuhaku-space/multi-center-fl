import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --- 設定 ---
LOG_DIR = Path("./logs")

st.set_page_config(
    page_title="FL Simulation Dashboard",
    page_icon="📊",
    layout="wide",
)


def load_experiments():
    """logsディレクトリ内の実験フォルダ一覧を取得"""
    if not LOG_DIR.exists():
        return []
    # 日付順（新しい順）にソート
    exps = sorted([d for d in LOG_DIR.iterdir() if d.is_dir()], reverse=True)
    return [d.name for d in exps]


@st.cache_data
def load_data(exp_id):
    """指定された実験IDのデータを読み込む"""
    exp_path = LOG_DIR / exp_id

    # 1. Config
    config = {}
    config_path = exp_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

    # 2. Log CSV (Metrics)
    log_df = pd.DataFrame()
    log_path = exp_path / "log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)

    # 3. Allocation CSV (Client Clustering)
    alloc_df = pd.DataFrame()
    alloc_path = exp_path / "client_allocation.csv"
    if alloc_path.exists():
        try:
            alloc_df = pd.read_csv(alloc_path)
        except pd.errors.EmptyDataError:
            pass

    return config, log_df, alloc_df


def main():
    st.title("📊 Federated Learning Visualization")

    # --- サイドバー: 実験選択 ---
    st.sidebar.header("Experiment Selection")
    experiments = load_experiments()

    if not experiments:
        st.warning("No experiments found in ./logs")
        return

    selected_exp = st.sidebar.selectbox("Select Experiment ID", experiments)

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()  # キャッシュクリアして再読み込み

    # データ読み込み
    config, log_df, alloc_df = load_data(selected_exp)

    # --- Config情報の表示 ---
    with st.expander("📝 Experiment Configuration", expanded=False):
        st.json(config)

    if log_df.empty:
        st.error("Log data (log.csv) is empty or not found.")
        return

    # --- メイン指標 (KPI) ---
    latest_log = log_df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Round", int(latest_log["Round"]))
    col2.metric("Accuracy", f"{latest_log['Accuracy']:.2f}%")
    col3.metric("Train Loss", f"{latest_log['Train Loss']:.4f}")
    col4.metric("Val Loss", f"{latest_log['Val Loss']:.4f}")

    # --- グラフ描画 ---
    st.divider()

    # 1. Loss & Accuracy 推移
    st.subheader("📈 Training Metrics")
    tab1, tab2 = st.tabs(["Accuracy & Loss", "Time per Round"])

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            fig_acc = px.line(
                log_df, x="Round", y="Accuracy", title="Global Accuracy", markers=True
            )
            # 修正: use_container_width=True -> width="stretch"
            st.plotly_chart(fig_acc, width="stretch")

        with c2:
            loss_cols = ["Train Loss", "Val Loss"]
            loss_df = log_df.melt(
                id_vars=["Round"],
                value_vars=loss_cols,
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
            # 修正: use_container_width=True -> width="stretch"
            st.plotly_chart(fig_loss, width="stretch")

    with tab2:
        if "Time" in log_df.columns:
            fig_time = px.bar(
                log_df, x="Round", y="Time", title="Execution Time per Round (seconds)"
            )
            # 修正: use_container_width=True -> width="stretch"
            st.plotly_chart(fig_time, width="stretch")
        else:
            st.info("Time data not available.")

    # 2. センターごとの所属人数推移
    st.subheader("👥 Cluster Sizes")
    center_cols = [c for c in log_df.columns if "Center_" in c]
    if center_cols:
        cluster_df = log_df.melt(
            id_vars=["Round"],
            value_vars=center_cols,
            var_name="Center",
            value_name="Count",
        )
        cluster_df["Center"] = cluster_df["Center"].apply(
            lambda x: x.replace("Center_", "C").replace("_Clients", "")
        )

        fig_cluster = px.area(
            cluster_df,
            x="Round",
            y="Count",
            color="Center",
            title="Client Distribution per Center",
        )
        # 修正: use_container_width=True -> width="stretch"
        st.plotly_chart(fig_cluster, width="stretch")

    # 3. クライアント所属ヒートマップ (改良版)
    num_centers = config.get("num_centers", 5)
    st.subheader("🗺️ Client Allocation Heatmap")
    if not alloc_df.empty:
        st.markdown("""
        **見方**: 縦軸が各ユーザー、横軸がラウンドの進行を表します。
        ユーザーは「最終ラウンドで所属しているセンター順」に自動ソートされています。
        色が安定して帯状になっている部分は、所属が固定化していることを示します。
        """)

        # データ整形
        heatmap_data = alloc_df.set_index("Round").T

        # ソート機能
        last_round = heatmap_data.columns[-1]
        heatmap_data = heatmap_data.sort_values(by=[last_round], ascending=True)

        # ページネーション
        total_writers = len(heatmap_data)
        if total_writers > 20:
            step = 20
            st.caption(
                f"Total Writers: {total_writers}. 表示範囲を選択してください（動作を軽くするため分割表示します）。"
            )
            start_idx = st.slider(
                "Display Range (Start Index)",
                min_value=0,
                max_value=total_writers - step,
                value=0,
                step=step,
            )
            end_idx = min(start_idx + step * 2, total_writers)
            heatmap_subset = heatmap_data.iloc[start_idx:end_idx]
        else:
            heatmap_subset = heatmap_data

        # 描画
        fig_heat = px.imshow(
            heatmap_subset,
            labels=dict(x="Round", y="Writer ID", color="Center ID"),
            aspect="auto",
            color_continuous_scale="Viridis",
            origin="lower",
        )

        fig_heat.update_traces(xgap=1, ygap=1)

        fig_heat.update_layout(
            height=600,
            xaxis_title="Round",
            yaxis_title="Writer ID (Sorted by Final Center)",
            coloraxis_colorbar=dict(
                title="Center ID",
                tickvals=list(range(num_centers)),
                ticktext=[f"C{i}" for i in range(num_centers)],
            ),
        )

        # 修正: use_container_width=True -> width="stretch"
        st.plotly_chart(fig_heat, width="stretch")

    else:
        st.info("No allocation data found (client_allocation.csv).")


if __name__ == "__main__":
    main()
