#
#
#
#
#           修改于2025.10.27，让公式里面必须包含t^-3/2这一项
#
#
#
#
#
#


from pysr import PySRRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.optimize import least_squares
import streamlit as st
import numpy as np
import sympy as sp


def symbolic_regression_model(data_pred):
    # ========= 配置 =========
    TOP_N = 10                # 展示前 N 条无约束候选
    SELECTION_MODE = 'index'  # 'index' | 'loss' | 'complexity'（仅用于展示时的高亮，不参与最终拟合）
    SELECT_IDX = 0
    LABEL_INDEX = [1, 2, 3, 4, 5]

    # ========= 数据准备 =========
    data_pred = data_pred.dropna().copy()
    data_pred = data_pred[data_pred['Predicted'].isin(LABEL_INDEX)].copy()
    data_pred = data_pred.sort_values(['Predicted', 'FallingTime(t/s)']).reset_index(drop=True)

    if (data_pred['FallingTime(t/s)'] <= 0).any():
        st.warning("警告：检测到 FallingTime(t/s)<=0。若包含 log/sqrt/负幂，可能导致数值问题，请考虑先对 FallingTime(t/s) 做平移或筛选。")

    # ========= 1) 无约束回归（仅展示候选） =========
    label_counts = data_pred['Predicted'].value_counts()
    max_label = label_counts.idxmax()
    sub_max_xy = data_pred[data_pred['Predicted'] == max_label]
    X_max = sub_max_xy[['FallingTime(t/s)']].values.astype(float)
    y_max = sub_max_xy['BalanceVoltage(U/V)'].values.astype(float)

    st.info(
        f"无约束符号回归展示：在样本最多的 label={max_label} 上回归 U=f(t)，"
        f"用于展示候选结构。"
    )

    model_display = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "exp"],
        model_selection="best",
        elementwise_loss="loss(x, y) = (x - y)^2",
        verbosity=0,
        maxsize=10,
        random_state=42,
        parallelism="serial",
        deterministic=True,
    )
    model_display.fit(X_max, y_max, variable_names=["t"])

    equations = model_display.equations_
    if equations is None or len(equations) == 0:
        st.error("无约束符号回归未生成任何候选公式，请调整 niterations 或检查数据。")
        # 即便如此，后面受约束拟合仍可继续
    else:
        equations = equations.sort_values(['loss', 'complexity'], ascending=[True, True]).reset_index(drop=True)
        show_n = min(TOP_N, len(equations))
        st.dataframe(equations[['loss', 'complexity', 'equation']].head(show_n))

        # 仅用于展示的“选中”高亮信息（不参与最终拟合）
        if SELECTION_MODE == 'index':
            pick_idx = SELECT_IDX
        elif SELECTION_MODE == 'loss':
            pick_idx = int(equations['loss'].idxmin())
        elif SELECTION_MODE == 'complexity':
            pick_idx = int(equations['complexity'].idxmin())
        else:
            pick_idx = 0

        if 0 <= pick_idx < len(equations):
            picked = equations.iloc[pick_idx]
            #st.info(
            #    "**展示用（无约束）候选的高亮：**\n\n"
            #    f"rank={pick_idx}, loss={picked['loss']:.6g}, complexity={int(picked['complexity'])}\n\n"
            #    f"equation：{picked['equation']}"
            #)
            # 保存展示用结果
            st.session_state['unconstrained_display'] = {
                "rank": int(pick_idx),
                "loss": float(picked['loss']),
                "complexity": int(picked['complexity']),
                "equation_str": str(picked['equation']),
                "sympy": picked['sympy_format'],
                "table_head": equations[['loss', 'complexity', 'equation']].head(show_n)
            }

    # ========= 2) 受约束最终结构：y(t) = a * t^(-3/2) + b =========
    st.info("最终拟合采用受约束统一结构： y(t) = a · t^(-3/2) + b")
    t = sp.Symbol('t', real=True, positive=True)
    a, b = sp.symbols('a b', real=True)
    expr_param = a * t**(-sp.Rational(3, 2)) + b
    params = [a, b]
    p0_vals = [1.0, 0.0]

    # 数值函数与数值安全包装
    g = sp.lambdify((t, *params), expr_param, modules='numpy')

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

    def fit_params_ab(t_arr_fit, y_arr_fit, p0):
        t_arr_fit = np.asarray(t_arr_fit, float)
        y_arr_fit = np.asarray(y_arr_fit, float)
        if (t_arr_fit <= 0).any():
            st.warning("检测到 FallingTime(t/s)<=0，拟合该组参数时对 t 做 +1e-9 微移以保证 t^(-3/2) 有意义。")
            t_arr_fit = t_arr_fit + 1e-9

        def residual(theta):
            yhat = g_checked(t_arr_fit, *theta)
            return yhat - y_arr_fit

        res = least_squares(
            residual,
            x0=np.asarray(p0, float),
            method='trf',
            loss='soft_l1',
            f_scale=1.0,
            max_nfev=20000,
        )
        if not res.success:
            res = least_squares(
                residual,
                x0=np.asarray(p0, float),
                method='trf',
                loss='linear',
                max_nfev=20000,
            )
        return res.x

    # ========= 3) 逐 label 拟合（受约束）并评估保存 =========
    st.session_state['regression_results'] = {
        'regression_form': "y(t) = a * t^(-3/2) + b",
        'evaluation': {},
        'data': {}
    }

    for label in LABEL_INDEX:
        sub = data_pred[data_pred['Predicted'] == label]
        if sub.empty:
            continue
        tt = sub['FallingTime(t/s)'].values.astype(float)
        yy = sub['BalanceVoltage(U/V)'].values.astype(float)

        try:
            popt = fit_params_ab(tt, yy, p0_vals)   # 拟合 [a, b]
        except FloatingPointError:
            tt_shift = tt + 1e-9
            popt = fit_params_ab(tt_shift, yy, p0_vals)

        # 评估
        y_pred = g_checked(tt, *popt)
        mse = mean_squared_error(yy, y_pred)
        mae = mean_absolute_error(yy, y_pred)
        r2 = r2_score(yy, y_pred)

        # 四舍五入后的表达式
        popt_rounded = [round(float(p), 4) for p in popt]
        fitted_expr = expr_param.subs({a: popt_rounded[0], b: popt_rounded[1]})

        # 保存
        st.session_state['regression_results']['evaluation'][label] = {
            "params": {"a": popt_rounded[0], "b": popt_rounded[1]},
            "mse": mse,
            "mae": mae,
            "r2": r2
        }

        t_line = np.linspace(tt.min(), tt.max(), 200)
        # 避免 t<=0 的绘图异常
        if np.any(t_line <= 0):
            pos_min = np.min(t_line[t_line > 0]) if np.any(t_line > 0) else 1e-9
            t_line = np.where(t_line <= 0, pos_min, t_line)
        y_line = g_checked(t_line, *popt)

        st.session_state['regression_results']['data'][label] = [t_line, y_line, fitted_expr]
