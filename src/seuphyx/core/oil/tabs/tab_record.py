"""
Tab 1: 数据记录
"""
# built-in
from pathlib import Path
# third-party
import streamlit as st
import pandas as pd
# seuphyx
from seuphyx.core.oil.utils import plotly_plot
import seuphyx


def render_tab_record():
    # user data file
    work_dir = st.session_state.work_dir
    oil_drop_csv = work_dir / "oil_drop.csv"

    # reference data file
    data_dir = Path(seuphyx.__file__).parent / "data"
    reference_file = data_dir / "oil_drop_reference.csv"

    # 记录数据表单
    with st.form(key="write_data", clear_on_submit=True):
        st.subheader("请你输入实验数据：")
        st.write("请你根据实验要求，输入下落时间 (t/s) 和 平衡电压 (U/V) 两个数据。")

        x_coords = [0.0, 0.0, 0.0, 0.0, 0.0]
        y_coords = [0.0, 0.0, 0.0, 0.0, 0.0]
        col1, col2, col3, col4, col5 = st.columns(5)
        for idx, col in enumerate((col1, col2, col3, col4, col5)):
            with col:
                st.write(f"**数据点 {idx+1}**")
                x_coords[idx] = st.number_input("下落时间 (t/s)",
                                                min_value=0.0,
                                                max_value=150.0,
                                                value=0.0,
                                                key=f"x_coord_{idx}")
                y_coords[idx] = st.number_input("平衡电压 (U/V)",
                                                min_value=0.0,
                                                max_value=400.0,
                                                value=0.0,
                                                key=f"y_coord_{idx}")

        submitted = st.form_submit_button("数据记录")
        if submitted:
            valid_coords = []
            for x_coord, y_coord in zip(x_coords, y_coords):
                if x_coord <= 0 or y_coord <= 0:
                    continue

                # 检查数据是否已存在（容差为1e-6）
                is_duplicate = False
                for _, row in st.session_state.data.iterrows():
                    if (abs(row['FallingTime(t/s)'] - x_coord) < 1e-6 and
                            abs(row['BalanceVoltage(U/V)'] - y_coord) < 1e-6):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    valid_coords.append((x_coord, y_coord))
                    st.sidebar.write(f"写入数据: ({x_coord}, {y_coord})")

            # 检查 oil_drop.csv 文件是否存在
            if not oil_drop_csv.exists():
                st.sidebar.warning(f"首次保存数据，已创建新文件 {oil_drop_csv} 。")
                with open(oil_drop_csv, "w") as file:
                    file.write("FallingTime(t/s),BalanceVoltage(U/V)\n")

            # 追加写入数据
            with open(oil_drop_csv, "a+") as file:
                for x_coord, y_coord in valid_coords:
                    file.write(f"{x_coord},{y_coord}\n")

            if len(valid_coords) != 0:
                st.session_state.data = pd.concat(
                    [
                        st.session_state.data,
                        pd.DataFrame(
                            valid_coords,
                            columns=[
                                "FallingTime(t/s)", "BalanceVoltage(U/V)"
                            ],
                        )
                    ],
                    ignore_index=True,
                )

            st.rerun()

    # 绘图描述
    with st.container(border=True):
        plotly_plot(
            title="实验数据散点图",
            grouped_data={
                "参考数据": st.session_state.data_ref.values,
                "实验数据": st.session_state.data.values,
            },
            key="scatter_plot",
            showlegend=True,
        )

        if st.button("**显示/隐藏参考数据**"):
            data_ref = pd.DataFrame(
                columns=["FallingTime(t/s)", "BalanceVoltage(U/V)"])
            if st.session_state.data_ref.empty:
                data_ref = pd.read_csv(reference_file)

            st.session_state.data_ref = data_ref
            st.rerun()
