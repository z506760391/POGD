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
    selected_feature_list = components.get('selected_feature_list', [])
    full_feature_list = components.get('full_feature_list', [])
    scaler = components.get('scaler')
    label_encoders = components.get('label_encoders', {})
    imputation_values = components.get('imputation_values', {})
    bagged_models = components.get('bagged_models')
    top_models = components.get('top_models_for_ensemble')
    ensemble_weights = components.get('ensemble_weights')

    if not all([selected_feature_list, full_feature_list, scaler, bagged_models, top_models, ensemble_weights]):
        st.error("模型文件 `ensemble_model_components.pkl` 已损坏或缺少关键组件。请重新运行主分析脚本。")
        st.stop()

    st.sidebar.header("患者临床信息输入")

    input_data = {}

    # --- 连续变量输入 ---
    age = st.sidebar.number_input("年龄 (Age)", min_value=18, max_value=100, value=60, help="患者必须为成年人 (≥18岁)")
    bmi = st.sidebar.number_input("体重指数 (BMI)", min_value=10.0, max_value=50.0, value=22.0, format="%.2f", help="范围: 10-50")
    blood = st.sidebar.number_input("术中出血量 (ml)", min_value=0, max_value=5000, value=100, help="范围: 0-5000ml")

    # --- 分类变量输入 ---
    node12_options = {'是 (Yes)': 1, '否 (No)': 0}
    node12_selection = st.sidebar.selectbox("病理淋巴结清扫数≥12 (node12)", options=list(node12_options.keys()), help="Harvested pathological lymph nodes ≥12")

    tmn_options = {
        "Stage 0 (Value 0)": 0,
        "Stage 0 (Value 1)": 1,
        "Stage Ⅰ (T1-2N0M0)": 2,
        "Stage Ⅱ (T3-4N0M0)": 3,
        "Stage Ⅲ (T1-4N1-2M0)": 4,
        "Stage Ⅳ (T1-4N0-2M1)": 5
    }
    tmn_selection = st.sidebar.selectbox("TMN分期 (TMN)", options=list(tmn_options.keys()), help="Tumour characteristics")

    add_options = {'是 (Yes)': 1, '否 (No)': 0}
    add_selection = st.sidebar.selectbox("额外器官切除 (Add)", options=list(add_options.keys()), help="Additional organ resection")

    dixon_miles_options = {'APR手术 (Abdominoperineal Resection)': 1, 'LAR手术 (Low Anterior Resection)': 0}
    dixon_miles_selection = st.sidebar.selectbox("手术方式 (DixonorRMiles)", options=list(dixon_miles_options.keys()))

    distant_group_options = {'≥7 cm': 1, '＜7 cm': 0}
    distant_group_selection = st.sidebar.selectbox("肿瘤距肛缘距离 (distant_group)", options=list(distant_group_options.keys()))

    nat_f_options = {'是 (Yes)': 1, '否 (No)': 0}
    nat_f_selection = st.sidebar.selectbox("新辅助治疗 (NAT_f)", options=list(nat_f_options.keys()), help="Neoadjuvant therapy")

    # 将用户选择的文本选项映射到模型需要的数值
    input_data = {
        'Age': age,
        'BMI': bmi,
        'blood': blood,
        'node12': node12_options[node12_selection],
        'TMN': tmn_options[tmn_selection],
        'Add': add_options[add_selection],
        'DixonorRMiles': dixon_miles_options[dixon_miles_selection],
        'distant_group': distant_group_options[distant_group_selection],
        'NAT_f': nat_f_options[nat_f_selection],
    }

    # "计算风险" 按钮
    if st.sidebar.button("计算风险", use_container_width=True, type="primary"):
        
        # --- 数据预处理 ---
        # 1. 将用户输入的 selected features 放入一个DataFrame
        input_df = pd.DataFrame([input_data])

        # 2. 构建包含所有原始特征的完整DataFrame，并用填充值补全
        full_df_data = {}
        for col in full_feature_list:
            if col in input_df.columns:
                full_df_data[col] = input_df[col].iloc[0]
            else:
                full_df_data[col] = imputation_values.get(col)
        full_df = pd.DataFrame([full_df_data])

        # 3. 对这个完整的DataFrame进行预处理
        # 3.1 编码分类变量
        for col, le in label_encoders.items():
            if col in full_df.columns:
                # 使用 .get() 来处理新类别
                full_df[col] = full_df[col].apply(lambda x: le.transform([x])[0] if str(x) in le.classes_ else le.transform(['__unk__'])[0])

        # 3.2 确保列顺序与训练时完全一致 (安全保障)
        full_df = full_df[full_feature_list]

        # 4. 标准化这个完整的DataFrame
        try:
            scaled_full_df = pd.DataFrame(scaler.transform(full_df), columns=full_feature_list)
            
            # 5. 从标准化的完整数据中，只选择模型需要的最终预测因子
            model_input_df = scaled_full_df[selected_feature_list]

            # --- 进行预测 ---
            prediction_proba = wsv(bagged_models, model_input_df, top_models, ensemble_weights)
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

            with st.expander("查看输入的详细数据 (最终预测因子)"):
                st.dataframe(input_df)
            with st.expander("查看为预处理构建的完整数据"):
                st.dataframe(full_df)
            with st.expander("查看模型最终的输入数据 (标准化后)"):
                st.dataframe(model_input_df)

        except Exception as e:
            st.error(f"在预测过程中发生错误: {e}")
            st.info("请确保所有输入值都是有效的。")

else:
    st.warning("模型组件未能加载，无法使用计算器功能。")

st.markdown("---")
st.info("""
**免责声明:** 本计算器仅用于学术研究和演示目的，其结果不应作为临床决策的唯一依据。所有临床决策都应由合格的医疗专业人员做出。
""")