from pysr import PySRRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.optimize import least_squares
import streamlit as st
import numpy as np
from sympy import Float, Symbol
import sympy as sp


def symbolic_regression_model(data_pred):
    # 用户可配置：候选展示与选择
    TOP_N = 10  # 展示前 N 条候选公式
    SELECTION_MODE = 'index'  # 'index' | 'loss' | 'complexity'
    SELECT_IDX = 0  # 当 SELECTION_MODE='index' 时有效（0 表示列表第一条）
    LABEL_INDEX = [1, 2, 3, 4, 5]

    # 读取外部数据
    data_pred = data_pred.dropna().copy()
    data_pred = data_pred[data_pred['Predicted'].isin(LABEL_INDEX)].copy()
    data_pred = data_pred.sort_values(['Predicted', 'FallingTime(t/s)'
                                       ]).reset_index(drop=True)

    if (data_pred['FallingTime(t/s)'] <= 0).any():
        st.warning("警告：检测到 FallingTime(t/s)<=0。"
                   "若统一结构包含 log/sqrt/负幂，"
                   "可能导致数值问题，"
                   "请考虑先对 FallingTime(t/s) 做平移或筛选。")

    # 1) 选样本最多的 label，符号回归抽取“候选结构库”
    label_counts = data_pred['Predicted'].value_counts()
    max_label = label_counts.idxmax()
    sub_max_xy = data_pred[data_pred['Predicted'] == max_label]
    X_max = sub_max_xy[['FallingTime(t/s)']].values
    y_max = sub_max_xy['BalanceVoltage(U/V)'].values

    st.info(f"数据点最多的类别值是: {max_label}\n\n"
            f"对 label = {max_label} 进行符号回归以抽取候选结构：")
    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "exp"],
        model_selection="best",  # 这里仍会有"best"，但我们下面改用 equations_ 列表来自行挑
        elementwise_loss="loss(x, y) = (x - y)^2",
        verbosity=0,
        maxsize=10,
        random_state=42,  # 固定随机种子，保证可复现
        parallelism="serial",
        deterministic=True,
    )
    model.fit(X_max, y_max, variable_names=["t"])

    # 所有候选（DataFrame），按 loss 升序，再按 complexity 升序
    equations = model.equations_
    if equations is None or len(equations) == 0:
        st.error("符号回归未生成任何候选公式，请调整 niterations 或检查数据。")

    equations = equations.sort_values(  # type: ignore
        ['loss', 'complexity'],
        ascending=[True, True],
    ).reset_index(drop=True)

    st.info("候选公式（按 loss、complexity 排序）")
    show_n = min(TOP_N, len(equations))
    st.dataframe(equations[['loss', 'complexity', 'equation']].head(show_n))

    # 选择规则
    if SELECTION_MODE == 'index':
        pick_idx = SELECT_IDX
    elif SELECTION_MODE == 'loss':
        pick_idx = int(equations['loss'].idxmin())
    elif SELECTION_MODE == 'complexity':
        pick_idx = int(equations['complexity'].idxmin())
    else:
        raise ValueError("SELECTION_MODE 必须是 'index' | 'loss' | 'complexity'")

    if not (0 <= pick_idx < len(equations)):
        raise IndexError(f"选择的索引 {pick_idx} 超出候选范围 0..{len(equations)-1}")

    picked = equations.iloc[pick_idx]

    # 打印最终选择的公式
    st.info("**最终选择的统一结构公式**\n\n"
            f"rank={pick_idx}, loss={picked['loss']:.6g}, "
            f"complexity={int(picked['complexity'])}\n\n"
            f"原始字符串公式：{picked['equation']}")

    # 取选中行对应的 SymPy 表达式
    # 注意：equations_ 的 'sympy_format' 列通常存的是 SymPy 表达式
    expr_raw = picked['sympy_format']
    st.session_state['expr_raw'] = expr_raw

    # ================= 2) 统一自变量名为 t，并把数值常数参数化 =================
    t = Symbol('t', real=True)

    # 把 Float 常数替换为参数 c1,c2,...，初值用原常数值
    params = []
    p0_vals = []

    def replace_floats(e):
        if isinstance(e, Float):
            p = sp.Symbol(f'c{len(params)+1}', real=True)
            params.append(p)
            p0_vals.append(float(e))
            return p
        if hasattr(e, 'args') and e.args:
            return e.func(*[replace_floats(arg) for arg in e.args])
        return e

    expr_param = replace_floats(expr_raw)

    # 若没有任何数值常数，兜底：整体缩放 + 偏置
    if len(params) == 0:
        c1, c2 = sp.symbols('c1 c2', real=True)
        expr_param = c1 * expr_param + c2
        params = [c1, c2]
        p0_vals = [1.0, 0.0]

    # print("\n统一结构（参数化）表达式：", expr_param)
    # print("参数列表：", [str(p) for p in params])
    # print("参数初值：", p0_vals)

    # 生成可数值函数 g(t, *theta)
    g = sp.lambdify((t, *params), expr_param, modules='numpy')

    # 数值安全包装
    def g_safe(tt, *theta, jitter=0.0):
        tt = np.asarray(tt, dtype=float)
        if jitter != 0.0:
            tt = tt + jitter
        val = g(tt, *theta)
        val = np.asarray(val)
        if val.dtype == object:
            val = val.astype(np.float64)
        return val

    def g_checked(tt, *theta):
        val = g_safe(tt, *theta, jitter=0.0)
        bad = ~np.isfinite(val)
        if np.any(bad):
            for eps in (1e-12, -1e-12, 1e-10, -1e-10):
                val2 = g_safe(tt, *theta, jitter=eps)
                if np.all(np.isfinite(val2)):
                    return val2.astype(np.float64)
            raise FloatingPointError("表达式在当前参数附近产生非有限值（可能分母接近0或log/sqrt域错误）。")
        return val.astype(np.float64)

    # 稳健拟合：least_squares + soft-L1
    def fit_params_for_group(t_arr, y_arr, p0):
        t_arr = np.asarray(t_arr, float)
        y_arr = np.asarray(y_arr, float)

        def residual(theta):
            yhat = g_checked(t_arr, *theta)
            return yhat - y_arr

        res = least_squares(
            residual,
            x0=np.asarray(p0, float),
            method='trf',
            loss='soft_l1',
            f_scale=1.0,
            max_nfev=20000,
        )
        if not res.success:
            # print("  警告：least_squares 未成功（", res.message, "），改用线性损失再试。")
            res = least_squares(
                residual,
                x0=np.asarray(p0, float),
                method='trf',
                loss='linear',
                max_nfev=20000,
            )
        return res.x

    #  3) 固定结构，仅拟合常数（逐 label）
    st.session_state['regression_results'] = {'evaluation': {}, 'data': {}}

    for label in LABEL_INDEX:
        sub = data_pred[data_pred['Predicted'] == label]
        tt = sub['FallingTime(t/s)'].values
        yy = sub['BalanceVoltage(U/V)'].values
        try:
            popt = fit_params_for_group(tt, yy, p0_vals)
        except FloatingPointError:
            tt_shift = tt + 1e-9
            popt = fit_params_for_group(tt_shift, yy, p0_vals)

        # 计算拟合结果并评估
        y_pred = g_checked(tt, *popt)
        mse = mean_squared_error(yy, y_pred)
        mae = mean_absolute_error(yy, y_pred)
        r2 = r2_score(yy, y_pred)

        # 生成该类别的拟合公式（将参数值代入，保留4位小数）
        # 先将参数值四舍五入到4位小数
        popt_rounded = [round(float(p), 4) for p in popt]
        fitted_expr = expr_param.subs(dict(zip(params, popt_rounded)))

        # 保存拟合参数和评估结果
        st.session_state['regression_results']['evaluation'][label] = {
            "params": dict(zip([str(p) for p in params], popt_rounded)),
            "mse": mse,
            "mae": mae,
            "r2": r2
        }

        t_line = np.linspace(tt.min(), tt.max(), 200)
        y_line = g_checked(t_line, *popt)

        # 保存拟合数据和公式表达式（使用四舍五入后的参数）
        st.session_state['regression_results']['data'][label] = [
            t_line, y_line, fitted_expr
        ]
