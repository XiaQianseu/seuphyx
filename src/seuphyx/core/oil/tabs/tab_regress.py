"""
Tab 3: 符号回归
"""
# built-in
from pathlib import Path

# third-party
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp

# seuphyx
from seuphyx.core.oil.tabs.regression import symbolic_regression_model
from seuphyx.core.oil.utils import plotly_plot
import seuphyx


def render_tab_regress():
    if 'data_pred' in st.session_state:
        with st.container(border=True):
            #
            data_dir = Path(seuphyx.__file__).parent / "data"
            reference_file = data_dir / "oil_drop_reference.csv"
            data_ref = pd.read_csv(reference_file)
            data = st.session_state.data
            data_combined = pd.concat([data, data_ref], axis=0)
            labels = st.session_state.model.predict(data_combined.values)

            # 显示结果
            y_pred_labels = np.unique(labels)
            grouped_data = {}
            for label in y_pred_labels:
                legend = f"舍弃数据" if label == y_pred_labels[-1] else f"类别{label}"
                grouped_data[legend] = data_combined[labels == label][[
                    'FallingTime(t/s)', 'BalanceVoltage(U/V)'
                ]].values

            # 绘制分类结果
            plotly_plot(
                title="分类实验数据散点图",
                grouped_data=grouped_data,
                key="classification_scatter_plot2",
                showlegend=True,
            )

            st.session_state.data_combined = pd.DataFrame({
                "FallingTime(t/s)":
                data_combined['FallingTime(t/s)'],
                "BalanceVoltage(U/V)":
                data_combined['BalanceVoltage(U/V)'],
                "Predicted":
                labels
            })

        with st.form("regression_form", border=False):
            submitted = st.form_submit_button("**对分类结果进行符号回归**")
            if submitted:
                symbolic_regression_model(st.session_state.data_combined)

        if 'expr_raw' in st.session_state:
            st.subheader("**符号回归得到的统一结构公式**")
            st.write(st.session_state.expr_raw)

        if 'regression_results' in st.session_state:
            st.subheader("**符号回归得到的统一结构公式**")
            fig = go.Figure(
                data=[
                    go.Scatter(x=data[:, 0],
                               y=data[:, 1],
                               mode="markers",
                               name=label,
                               showlegend=True)
                    for label, data in grouped_data.items()
                ] + [
                    go.Line(x=t_line, y=y_line, name=f'符号回归-类别{label}')
                    for label, (t_line, y_line, fitted_expr) in
                    st.session_state['regression_results']['data'].items()
                ],
                layout=go.Layout(
                    xaxis=dict(title='下落时间 (t/s)'),
                    yaxis=dict(title='平衡电压 (U/V)'),
                    font=dict(family='DejaVu Serif', size=16),
                    margin=dict(l=60, r=30, t=30, b=60),
                    colorway=px.colors.qualitative.D3,
                ),
            )
            st.plotly_chart(fig, key="regression_plot")

            for label, (_, _, fitted_expr) in st.session_state[
                    'regression_results']['data'].items():
                st.write(f"**类别 {label} 拟合公式:**", fitted_expr)
                # st.latex(sp.latex(fitted_expr))

