import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===========================================================
# 页面配置
# ===========================================================
st.set_page_config(
    page_title="POGD 风险预测计算器",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===========================================================
# 辅助函数 (从主脚本中复制并简化)
# ===========================================================
def agg_pred(bag_mods, X, nm):
    pl = []
    for bd in bag_mods:
        m = bd.get(nm)
        if m:
            try:
                pl.append(m.predict_proba(X)[:, 1])
            except Exception:
                pass
    return np.mean(pl, axis=0) if pl else np.zeros(len(X))


def wsv(bag_mods, X, tops, weights_dict):
    pl, ws = [], []
    for n in tops:
        w = weights_dict.get(n, 0.)
        ws.append(0. if (isinstance(w, float) and np.isnan(w)) else float(w))
        pl.append(agg_pred(bag_mods, X, n))
    ws = np.array(ws, dtype=float)
    ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(ws)) / len(ws)
    return np.dot(ws, np.array(pl))


# ===========================================================
# 加载模型和组件
# ===========================================================
@st.cache_resource
def load_model_components():
    """
    加载所有必要的模型组件。
    使用 Streamlit 的缓存来避免每次交互都重新加载。
    """
    path = 'output/ensemble_model_components.pkl'
    if not os.path.exists(path):
        st.error(f"错误：找不到模型文件 '{path}'。请先运行主分析脚本 `ml_pogd_final_v5_1 - 副本.py` 来生成模型文件。")
        return None
    try:
        components = joblib.load(path)
        return components
    except Exception as e:
        st.error(f"加载模型文件时出错: {e}")
        return None


components = load_model_components()

# ===========================================================
# 主应用界面
# ===========================================================
st.title("⚕️ POGD 术后胃功能障碍风险预测计算器")
st.markdown("---")
st.markdown("""
本工具根据已发表的预测模型，计算患者术后发生胃功能障碍（POGD）的风险。
请在下方输入患者的临床指标。
""")

if components:
    # 从加载的组件中提取所需对象
    feature_list = components.get('feature_list', [])
    scaler = components.get('scaler')
    label_encoders = components.get('label_encoders', {})
    imputation_values = components.get('imputation_values', {})
    bagged_models = components.get('bagged_models')
    top_models = components.get('top_models_for_ensemble')
    ensemble_weights = components.get('ensemble_weights')

    st.sidebar.header("患者指标输入")

    input_data = {}

    # 创建输入字段
    for feature in feature_list:
        # 检查是否为分类变量
        if feature in label_encoders:
            le = label_encoders[feature]
            # 移除'__unk__'和'__m__'
            options = [cls for cls in le.classes_ if cls not in ['__unk__', '__m__']]
            input_data[feature] = st.sidebar.selectbox(f"**{feature}**", options, help=f"选择 {feature} 的值")
        else:
            # 对于数值型变量
            default_val = float(imputation_values.get(feature, 0.0))
            input_data[feature] = st.sidebar.number_input(
                f"**{feature}**",
                value=default_val,
                format="%.2f",
                help=f"输入 {feature} 的数值。默认值为训练集的中位数/众数 ({default_val:.2f})。"
            )

    # 创建一个用于预测的DataFrame
    input_df = pd.DataFrame([input_data])

    # "计算风险" 按钮
    if st.sidebar.button("计算风险", use_container_width=True, type="primary"):

        # --- 数据预处理 ---
        # 1. 编码分类变量
        for col, le in label_encoders.items():
            if col in input_df.columns:
                # 使用 .get() 来处理新类别
                input_df[col] = input_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(['__unk__'])[0])

        # 2. 确保所有特征列都存在，并按正确顺序排列
        processed_df = pd.DataFrame(columns=feature_list)
        processed_df = pd.concat([processed_df, input_df], ignore_index=True)

        # 3. 填充在UI中可能产生的任何缺失值 (理论上不会)
        for col in feature_list:
            if col not in processed_df.columns or processed_df[col].isnull().any():
                processed_df[col] = processed_df[col].fillna(imputation_values[col])

        # 4. 标准化
        try:
            scaled_df = pd.DataFrame(scaler.transform(processed_df[feature_list]), columns=feature_list)

            # --- 进行预测 ---
            prediction_proba = wsv(bagged_models, scaled_df, top_models, ensemble_weights)
            risk_percentage = prediction_proba[0] * 100

            # --- 显示结果 ---
            st.header("预测结果")
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="POGD 风险概率",
                    value=f"{risk_percentage:.2f} %"
                )

            with col2:
                if risk_percentage >= 50:
                    st.error("结论：高风险")
                elif risk_percentage >= 30:
                    st.warning("结论：中等风险")
                else:
                    st.success("结论：低风险")

            st.progress(risk_percentage / 100)

            with st.expander("查看输入的详细数据"):
                st.dataframe(input_df)
            with st.expander("查看预处理后的数据 (模型输入)"):
                st.dataframe(scaled_df)

        except Exception as e:
            st.error(f"在预测过程中发生错误: {e}")
            st.info("请确保所有输入值都是有效的。")

else:
    st.warning("模型组件未能加载，无法使用计算器功能。")

st.markdown("---")
st.info("""
**免责声明:** 本计算器仅用于学术研究和演示目的，其结果不应作为临床决策的唯一依据。所有临床决策都应由合格的医疗专业人员做出。
""")
