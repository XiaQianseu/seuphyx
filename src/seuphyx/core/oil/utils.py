"""
通用工具函数模块
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


def plotly_plot(title, grouped_data, key, showlegend):
    """
    绘制 Plotly 散点图
    
    Args:
        title: 图表标题
        grouped_data: 字典，格式为 {label: data_array}
        key: Streamlit 组件的唯一 key
        showlegend: 是否显示图例
    """
    st.subheader(title)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=data[:, 0],
                y=data[:, 1],
                mode="markers",
                name=label,
                showlegend=showlegend
            )
            for label, data in grouped_data.items()
        ],
        layout=go.Layout(
            xaxis=dict(title='下落时间 (t/s)', range=[0, 85]),
            yaxis=dict(title='平衡电压 (U/V)', range=[0, 360]),

            font=dict(family='DejaVu Serif', size=16),
            margin=dict(l=60, r=30, t=30, b=60),
            colorway=px.colors.qualitative.D3,
        ),
    )
    st.plotly_chart(fig, key=key)
