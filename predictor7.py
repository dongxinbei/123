# ==================== 导入核心库 ====================
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 基础配置 ====================
# 加载训练好的模型（确保svm.pkl与脚本同目录）
model = joblib.load('svm.pkl')

# 加载测试数据（用于LIME解释器，确保X_test.csv与脚本同目录）
X_test = pd.read_csv('X_test.csv')

# 定义特征名称（替换为业务相关列名，与编码规则对应）
feature_names = [
    "Type_of_anesthesia", "Diabetes", "Transfusion", "Operate_time", 
    "BMI", "Waiting_time", "Hb", "Platelet",
]

# ==================== 2. Streamlit页面配置 ====================
st.set_page_config(page_title="TransfuseAI", layout="wide")
st.title("TransfuseAI")
st.markdown('Please fill in the following information and click "Predict" to obtain the Anemia risk assessment result')

# ==================== 3. 特征输入组件（按编码规则设计） ====================
# 1. 麻醉方式（0：腰麻，1：全麻）
Type_of_anesthesia = st.selectbox(
    "Type_of_anesthesia",
    options=[0, 1],
    format_func=lambda x: "Combined Spinal-Epidural Anesthesia" if x == 0 else "General Anesthesia"
)

# 2. 糖尿病病史（0：正常，1：异常）
Diabetes = st.selectbox(
    "History_of_Diabetes_Mellitus",
    options=[0, 1],
    format_func=lambda x: "NO" if x == 0 else "YES"
)

# 3. 体重指数BMI（连续变量，保留1位小数）
BMI = st.number_input(
    "BMI(kg/m²)",
    min_value=5.0,
    max_value=50.0,
    value=22.0,
    step=0.1,
    format="%.1f"
)

# 4. 手术时间（连续变量，单位：分钟）
Operate_time = st.slider(
    "operate_time（min）",
    min_value=5,
    max_value=600,
    value=120,
    step=10
)

# 5. 术前血红蛋白（连续变量，单位：g/L）
Hb = st.number_input(
    "preop_hemoglobin（g/L）",
    min_value=30.0,
    max_value=200.0,
    value=130.0,
    step=1.0,
    format="%.1f"
)

# 6. 术前血小板（连续变量，单位：g/L）
Platelet = st.number_input(
    "Platelet（bil/L）",
    min_value=0.0,
    max_value=20000.0,
    value=130.0,
    step=1.0,
    format="%.1f"
)

# 7. 等待手术时间（连续变量，单位：天）
Waiting_time = st.slider(
    "waiting_time（d）",
    min_value=0,
    max_value=60,
    value=3,
    step=1
)

# 8. 备血量（连续变量，单位：ml）
Transfusion = st.slider(
    "Blood_transfusion_volume（ml）",
    min_value=0,
    max_value=5000,
    value=3,
    step=1
)

# ==================== 4. 数据处理与预测 ====================
# 整合用户输入特征（注意：图片中只显示了2个变量，实际需要补充完整）
feature_values = [
    Type_of_anesthesia, Diabetes, Transfusion, Operate_time, 
    BMI, Waiting_time, Hb, Platelet, # 已显示的部分
    # 以下变量在图片中提及但未显示完整输入组件，需要补充：
    # psychological_counseling, handrail, multiple_drugs,
    # safety_warning, hospitalization, economy, phq, exercise_times,
    # acezong, education, fitness_area, childhood_health, childhood_economy
]

# 转换为模型输入格式
features = np.array([feature_values])

# 预测按钮逻辑
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]  # 0: 低风险, 1: 高风险
    predicted_proba = model.predict_proba(features)[0]  # 概率值
    
    # 显示预测结果（中文适配）
    st.subheader("Prediction_Result")
    risk_label = "high risk" if predicted_class == 1 else "lower risk"
    st.write(f"**Risk Level：{predicted_class} ({risk_label}) **")
    st.write(f"**Risk Probability：** Low-risk probability {predicted_proba[0]:.2%} | high-risk probability {predicted_proba[1]:.2%}")
    
    # 生成个性化建议（中文）
    st.subheader("💡 Recommendation")
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"The model predicts that the patient is at extremely high risk of developing postoperative anemia.（probability{probability:.1f}%）。"
            "It is recommended to increase intraoperative blood reserve before surgery, prioritize combined spinal-epidural anesthesia (CSEA) in the absence of contraindications, select senior surgeons experienced with PFNA procedure."
            "closely monitor postoperative blood routine indicators."
        )
    if predicted_class ==0:
        advice = (
            f"closely monitor postoperative blood routine indicators."
        )
        
        st.write(advice)
        
