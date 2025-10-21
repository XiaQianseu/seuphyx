"""
Tab 4: 打印报告
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ..utils import plotly_plot

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import matplotlib.pyplot as plt
# 创建 PDF 文件的函数
def save_to_pdf():
    # 获取保存路径
    work_dir = st.session_state.work_dir
    pdf_file = work_dir / "regression_results.pdf"
    
    # 创建 PDF 画布
    c = canvas.Canvas(str(pdf_file), pagesize=letter)
    
    # 设置字体和大小
    c.setFont("Helvetica", 12)
    
    # 获取回归结果
    if 'regression_results' in st.session_state:
        data = st.session_state['regression_results']['data'].items()
        for label, (_, _, fitted_expr) in data:
            st.write(f"**类别 {label} 拟合公式:**", fitted_expr)
        data = st.session_state['regression_results']['data'].items()
        y_position = 750  # 初始位置，用于控制文本高度
        
        # 将每个类别和公式保存到 PDF
        for label, (_, _, fitted_expr) in data:
            # 写入类别和拟合公式
            c.drawString(72, y_position, f"类别 {label} 拟合公式: {fitted_expr}")
            y_position -= 20  # 每写一行文本，向下移动
            
            # 如果公式较长，可能需要换行处理（此处可以根据需要调整）
            if y_position < 100:
                c.showPage()  # 创建新页面
                c.setFont("Helvetica", 12)
                y_position = 750  # 重置位置
            
        c.save()  # 保存 PDF 文件
        st.sidebar.write(f"回归结果已保存到: {pdf_file}")

def render_tab_report(work_dir, student_id, student_name):
    """
    渲染打印报告页面
    
    Args:
        work_dir: 工作目录路径
        student_id: 学生学号
        student_name: 学生姓名
    """
    # 绘图描述
    if 'data_pred' in st.session_state:
        y_pred_labels = np.unique(st.session_state.data_pred['Predicted'])
        grouped_data = {}
        for label in y_pred_labels:
            grouped_data[f"类别{label}"] = st.session_state.data_pred[
                st.session_state.data_pred['Predicted'] == label][[
                    'FallingTime(t/s)', 'BalanceVoltage(U/V)'
                ]].values

        with st.container(border=True):
            # if 'expr_raw' in st.session_state:
            #     expr_raw = st.session_state.expr_raw
            #     st.subheader("**符号回归得到的统一结构公式**")
            #     st.latex(rf"{expr_raw}")

            # if 'regression_results' in st.session_state:
            #     data = st.session_state['regression_results']['data'].items()
            #     for label, (_, _, fitted_expr) in data:
            #         st.write(f"**类别 {label} 拟合公式:**", fitted_expr)

#============================================================================================================            
#                                             保存文本     夏谦修改2025.10.21                              #=
#============================================================================================================
            # 获取保存路径                                                                                 #=
            work_dir = st.session_state.work_dir                                                           #=
            pdf_file = work_dir / "regression_results.pdf"                                                 #=
                                                                                                           #=
            # 创建 PDF 画布                                                                                #=
            c = canvas.Canvas(str(pdf_file), pagesize=letter)                                              #=
                                                                                                           #=
            # import os                                                                                    #=
            # print(os.path.abspath('./Ubuntu_18.04_SimHei.ttf'))                                          #=
            # # 注册中文字体 (字体文件与代码在同一目录下)                                                  #=
            font_path = "./SimHei.ttf"  # 字体文件的相对路径                                               #=
            pdfmetrics.registerFont(TTFont('SimHei', font_path))  # 注册字体                               #=
                                                                                                           #=
                                                                                                           #=
            # 设置字体和大小                                                                               #=
            c.setFont("SimHei", 12)                                                                        #=
                                                                                                           #=
            # 获取回归结果                                                                                 #=
            if 'regression_results' in st.session_state:                                                   #=
                data = st.session_state['regression_results']['data'].items()                              #=
                for label, (_, _, fitted_expr) in data:                                                    #=
                    st.write(f"**类别 {label} 拟合公式:**", fitted_expr)                                   #=
                data = st.session_state['regression_results']['data'].items()                              #=
                y_position = 750  # 初始位置，用于控制文本高度                                             #=
                                                                                                           #=
                # 将每个类别和公式保存到 PDF                                                               #=
                for label, (_, _, fitted_expr) in data:                                                    #=
                    # 写入类别和拟合公式                                                                   #=
                    c.drawString(72, y_position, f"类别 {label} 拟合公式: {fitted_expr}")                  #=
                    y_position -= 20  # 每写一行文本，向下移动                                             #=
                                                                                                           #=
                    # 如果公式较长，可能需要换行处理（此处可以根据需要调整）                               #=
                    if y_position < 100:                                                                   #=
                        c.showPage()  # 创建新页面                                                         #=
                        c.setFont("Helvetica", 12)                                                         #=
                        y_position = 750  # 重置位置                                                       #=
                                                                                                           #=
                c.save()  # 保存 PDF 文件                                                                  #=
                st.sidebar.write(f"回归结果已保存到: {pdf_file}")                                          #=
#============================================================================================================

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
                st.plotly_chart(fig, key="regression_plot_print")

#============================================================================================================            
#                                             保存图片     夏谦修改2025.10.21                              #=
#============================================================================================================
                # 保存图像到文件                                                                           #=
                work_dir = st.session_state.work_dir  # 获取保存路径                                       #=
                image_file = work_dir / "regression_plot.png"  # 设置保存图像的文件路径                    #=
                fig.write_image(str(image_file))  # 保存图像为 PNG 文件                                    #=
                st.sidebar.write(f"图像已保存至: {image_file}")                                            #=
#============================================================================================================

        with st.container(border=True):
            st.subheader("已保存的数据点：")
            st.dataframe(
                st.session_state.data_pred,
                on_select="ignore",
                height=35 * len(st.session_state.data_pred) + 38,
            )
#============================================================================================================            
#                                             保存表格     夏谦修改2025.10.21                              #=
#============================================================================================================
            import os                                                                                      #=
            from pandas.plotting import table                                                              #=
            data = st.session_state.data_pred  # 你的 DataFrame 数据                                       #=
                                                                                                           #=
            # 获取保存路径                                                                                 #=
            work_dir = st.session_state.work_dir                                                           #=
            image_path = os.path.join(work_dir, "data_table.png")                                          #=
                                                                                                           #=
            # 根据列数动态调整表格宽度（列多时更宽）                                                       #=
            ncols = len(data.columns)                                                                      #=
            fig_width = max(8, ncols * 1.2)   # 每列约1.2英寸，至少8英寸宽                                 #=
            fig_height = min(0.6 * len(data) + 2, 15)  # 高度随行数变化，最多15英寸                        #=
                                                                                                           #=
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))                                        #=
            ax.axis('off')                                                                                 #=
                                                                                                           #=
            # 绘制表格，字体略大一点                                                                       #=
            tbl = table(                                                                                   #=
                ax, data, loc='center', cellLoc='center',                                                  #=
                colWidths=[1.0 / ncols] * ncols                                                            #=
            )                                                                                              #=
            tbl.auto_set_font_size(False)                                                                  #=
            tbl.set_fontsize(10)   # 调大字体                                                              #=
            tbl.scale(1.2, 1.2)    # 整体放大表格比例                                                      #=
                                                                                                           #=
            # 保存为高分辨率图像                                                                           #=
            plt.savefig(image_path, bbox_inches='tight', dpi=400)  # 提高清晰度                            #=
            plt.close()                                                                                    #=
#============================================================================================================

        st.markdown("**请你保存本页面内容，作为本次实验的报告提交。**")

#============================================================================================================            
#                                             保存表格     夏谦修改2025.10.21                              #=
#============================================================================================================
        from io import BytesIO                                                                             #=
        from reportlab.lib.utils import ImageReader                                                        #=
        from pathlib import Path                                                                           #=
        # —— 在 fig.write_image(...) 与保存 data_table.png 之后插入 ——                                     #=
        with st.container(border=True):                                                                    #=
            st.markdown("### 生成并下载报告 PDF")                                                          #=
            if st.button("生成并下载报告 PDF", use_container_width=True):                                  #=
                work_dir = st.session_state.work_dir                                                       #=
                pdf_path = work_dir / "regression_results.pdf"                                             #=
                                                                                                           #=
                # 1) 取回回归结果（用于把公式写入 PDF）                                                    #=
                reg_items = list(st.session_state['regression_results']['data'].items())                   #=
                                                                                                           #=
                # 2) 内存里先生成 PDF（便于立刻下载），同时也保存到硬盘                                    #=
                buf = BytesIO()                                                                            #=
                c = canvas.Canvas(buf, pagesize=letter)                                                    #=
                                                                                                           #=
                # 2.1 字体设置：优先用 SimHei（若不存在则回退 Helvetica）                                  #=
                font_name = "SimHei"                                                                       #=
                # font_file = Path("./SimHei.ttf")                                                         #=
                # try:                                                                                     #=
                #     if font_file.exists():                                                               #=
                #         pdfmetrics.registerFont(TTFont(font_name, str(font_file)))                       #=
                #     else:                                                                                #=
                #         font_name = "Helvetica"                                                          #=
                # except Exception:                                                                        #=
                #     font_name = "Helvetica"                                                              #=
                # c.setFont(font_name, 12)                                                                 #=
                font_path = "./SimHei.ttf"  # 字体文件的相对路径                                           #=
                pdfmetrics.registerFont(TTFont('SimHei', font_path))  # 注册字体                           #=
                                                                                                           #=
                                                                                                           #=
                # 设置字体和大小                                                                           #=
                c.setFont("SimHei", 12)                                                                    #=
                                                                                                           #=
                                                                                                           #=
                # 2.2 先写拟合公式（与页面一致）                                                           #=
                y = 750                                                                                    #=
                for label, (_, _, expr) in reg_items:                                                      #=
                    c.drawString(72, y, f"类别 {label} 拟合公式: {expr}")                                  #=
                    y -= 20                                                                                #=
                    if y < 100:                                                                            #=
                        c.showPage()                                                                       #=
                        c.setFont(font_name, 12)                                                           #=
                        y = 750                                                                            #=
                                                                                                           #=
                # 2.3 把两张图片依次追加到 PDF 尾部（自动等比缩放并居中）                                  #=
                ###########################################                                                #=
                # 2.3 把两张图片尽量排在同一页（上下紧贴，放不下再换页）                                   #=
                page_w, page_h = letter                                                                    #=
                margin = 36                                                                                #=
                gap = 12                                                                                   #=
                max_w = page_w - 2 * margin                                                                #=
                max_page_h = page_h - 2 * margin                                                           #=
                                                                                                           #=
                # 若公式写完后剩余空间太小，先换页再开始贴图                                               #=
                if y < margin + 50:                                                                        #=
                    c.showPage()                                                                           #=
                    c.setFont(font_name, 12)                                                               #=
                    y = page_h - margin                                                                    #=
                                                                                                           #=
                image_paths = [                                                                            #=
                    work_dir / "regression_plot.png",                                                      #=
                    work_dir / "data_table.png",                                                           #=
                ]                                                                                          #=
                images = []                                                                                #=
                for p in image_paths:                                                                      #=
                    if p.exists():                                                                         #=
                        img = ImageReader(str(p))                                                          #=
                        iw, ih = img.getSize()                                                             #=
                        images.append((img, iw, ih))                                                       #=
                                                                                                           #=
                def draw_images_tightly(imgs, start_y):                                                    #=
                    y_cursor = start_y                                                                     #=
                    idx = 0                                                                                #=
                    while idx < len(imgs):                                                                 #=
                        avail_h = y_cursor - margin                                                        #=
                        if avail_h <= 60:                                                                  #=
                            c.showPage()                                                                   #=
                            c.setFont(font_name, 12)                                                       #=
                            y_cursor = page_h - margin                                                     #=
                            avail_h = y_cursor - margin                                                    #=
                                                                                                           #=
                        pack = []                                                                          #=
                        total_h_base = 0.0                                                                 #=
                        for j in range(idx, len(imgs)):                                                    #=
                            _, iw, ih = imgs[j]                                                            #=
                            base_scale = min(max_w / iw, max_page_h / ih)                                  #=
                            base_h = ih * base_scale                                                       #=
                            if len(pack) == 0 or (total_h_base + (gap if pack else 0) + base_h) <= avail_h:#=
                                pack.append((j, base_scale, base_h))                                       #=
                                total_h_base += (gap if len(pack) > 1 else 0) + base_h                     #=
                            else:                                                                          #=
                                break                                                                      #=
                                                                                                           #=
                        if not pack:                                                                       #=
                            j, iw, ih = idx, imgs[idx][1], imgs[idx][2]                                    #=
                            img = imgs[idx][0]                                                             #=
                            scale = min(max_w / iw, avail_h / ih)                                          #=
                            draw_w, draw_h = iw * scale, ih * scale                                        #=
                            x = (page_w - draw_w) / 2                                                      #=
                            c.drawImage(img, x, y_cursor - draw_h,                                         #=
                                        width=draw_w, height=draw_h,                                       #=
                                        preserveAspectRatio=True, mask='auto')                             #=
                            y_cursor -= (draw_h + gap)                                                     #=
                            idx += 1                                                                       #=
                            continue                                                                       #=
                                                                                                           #=
                        total_h = sum(h for _, _, h in pack) + gap * (len(pack) - 1)                       #=
                        shrink = min(1.0, avail_h / total_h) if total_h > avail_h else 1.0                 #=
                                                                                                           #=
                        for j, base_scale, base_h in pack:                                                 #=
                            img, iw, ih = imgs[j]                                                          #=
                            scale = base_scale * shrink                                                    #=
                            draw_w, draw_h = iw * scale, ih * scale                                        #=
                            x = (page_w - draw_w) / 2                                                      #=
                            c.drawImage(img, x, y_cursor - draw_h,                                         #=
                                        width=draw_w, height=draw_h,                                       #=
                                        preserveAspectRatio=True, mask='auto')                             #=
                            y_cursor -= (draw_h + gap)                                                     #=
                                                                                                           #=
                        idx = pack[-1][0] + 1                                                              #=
                                                                                                           #=
                    return y_cursor                                                                        #=
                                                                                                           #=
                y = draw_images_tightly(images, y)                                                         #=
        ########################################                                                           #=
                                                                                                           #=
                c.save()                                                                                   #=
                pdf_bytes = buf.getvalue()                                                                 #=
                buf.close()                                                                                #=
                                                                                                           #=
                # 同时落地保存一份                                                                         #=
                with open(pdf_path, "wb") as f:                                                            #=
                    f.write(pdf_bytes)                                                                     #=
                                                                                                           #=
                st.sidebar.success(f"报告已生成：{pdf_path}")                                              #=
                st.download_button(                                                                        #=
                    label="下载报告 PDF",                                                                  #=
                    data=pdf_bytes,                                                                        #=
                    file_name="regression_results.pdf",                                                    #=
                    mime="application/pdf",                                                                #=
                    use_container_width=True,                                                              #=
                )                                                                                          #=
#============================================================================================================