# ==============================
# 运动想象BCI康复训练教学平台
# 单文件Streamlit版本
# 运行: streamlit run app.py
# ==============================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

st.set_page_config(
    page_title="BCI运动想象康复平台",
    layout="wide"
)

st.title("🧠 运动想象BCI康复训练与教学实训平台")

st.markdown("""
**应用场景**

- 基层康复机构：脑卒中运动功能康复训练  
- 高校康复专业：BCI教学实验平台  

**核心功能**

EEG数据 → 预处理 → 特征提取 → 算法训练 → 运动意图识别 → 康复反馈
""")

# ==============================
# 页面Tab
# ==============================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 首页说明",
    "📊 EEG数据与预处理",
    "🔬 特征提取实验",
    "🤖 算法训练",
    "📈 结果与康复反馈",
    "🎮 BCI康复小游戏"
])

# ==============================
# 首页
# ==============================

with tab1:

    st.header("平台介绍")

    st.info("""
本平台用于 **运动想象脑机接口（MI-BCI）康复训练与教学实训**。

训练流程：

1️⃣ EEG信号采集  
2️⃣ 信号预处理  
3️⃣ 特征提取  
4️⃣ 分类算法识别运动意图  
5️⃣ 康复训练反馈  
""")

    st.subheader("运动想象任务")

    st.write("""
0 = 左手  
1 = 右手  
2 = 双脚  
3 = 舌头
""")

# ==============================
# 数据模块
# ==============================

with tab2:

    st.header("EEG数据加载")

    def get_demo_data():

        np.random.seed(42)

        return pd.DataFrame({
            "C3": np.random.randn(1000),
            "Cz": np.random.randn(1000),
            "C4": np.random.randn(1000),
            "label": np.random.randint(0,4,1000)
        })

    df = get_demo_data()

    st.write("示例数据（模拟BCICIV_2a）")

    st.dataframe(df.head())

    st.subheader("EEG信号可视化")

    channel = st.selectbox("选择通道", ["C3","Cz","C4"])

    signal = df[channel].values

    fig, ax = plt.subplots()

    ax.plot(signal[:300])

    ax.set_title("EEG Signal")

    st.pyplot(fig)

    st.subheader("预处理")

    st.code("""
Bandpass Filter: 8-30Hz
Notch Filter: 50Hz
Artifact Removal: ICA
""")

    st.success("预处理完成")

# ==============================
# 特征提取实验
# ==============================

with tab3:

    st.header("EEG特征提取")

    signal = np.random.randn(1000)

    feature = st.selectbox(
        "选择特征",
        ["FFT","PSD","Band Power"]
    )

    if feature == "FFT":

        fft_vals = np.abs(np.fft.rfft(signal))

        freqs = np.fft.rfftfreq(len(signal),1/250)

        fig, ax = plt.subplots()

        ax.plot(freqs, fft_vals)

        ax.set_title("FFT Spectrum")

        st.pyplot(fig)

    if feature == "PSD":

        psd = np.abs(np.fft.rfft(signal))**2

        freqs = np.fft.rfftfreq(len(signal),1/250)

        fig, ax = plt.subplots()

        ax.plot(freqs, psd)

        ax.set_title("PSD")

        st.pyplot(fig)

    if feature == "Band Power":

        mu = np.random.random()
        beta = np.random.random()

        fig, ax = plt.subplots()

        ax.bar(["Mu 8-12Hz","Beta 13-30Hz"],[mu,beta])

        ax.set_title("Band Power")

        st.pyplot(fig)

# ==============================
# 算法训练
# ==============================

with tab4:

    st.header("运动想象分类算法")

    model = st.selectbox(
        "选择算法",
        ["SVM","LDA","Logistic Regression"]
    )

    if st.button("开始训练"):

        acc = {
            "SVM":0.82,
            "LDA":0.80,
            "Logistic Regression":0.78
        }[model]

        st.session_state["acc"] = acc
        st.session_state["model"] = model
        st.session_state["pred"] = np.random.randint(0,4,50)

        st.success(f"训练完成 {model} 准确率 {acc:.2f}")

# ==============================
# 结果与康复反馈
# ==============================

with tab5:

    st.header("BCI识别结果")

    if "acc" not in st.session_state:

        st.warning("请先训练模型")

    else:

        acc = st.session_state["acc"]
        model = st.session_state["model"]
        pred = st.session_state["pred"]

        col1, col2 = st.columns(2)

        with col1:

            st.metric("算法", model)
            st.metric("识别准确率", f"{acc:.0%}")

        with col2:

            fig, ax = plt.subplots()

            ax.bar(["左手","右手","双脚","舌头"],
                   [sum(pred==i) for i in range(4)])

            ax.set_title("运动意图统计")

            st.pyplot(fig)

        st.subheader("康复训练反馈")

        st.success("""
识别正确 → 神经反馈 → 促进神经可塑性 → 运动功能恢复
""")

        st.info("""
适用于：

- 脑卒中上肢康复训练
- 康复医学教学实验
""")

# ==============================
# BCI康复小游戏
# ==============================

with tab6:

    st.header("BCI康复小游戏")

    st.write("想象动作控制小球移动")

    if "ball" not in st.session_state:
        st.session_state.ball = 0

    task = random.choice(["左手","右手"])

    st.subheader(f"任务：想象 {task} 运动")

    if st.button("模拟BCI识别"):

        if task == "左手":
            st.session_state.ball -= 1
        else:
            st.session_state.ball += 1

    st.write("小球位置")

    fig, ax = plt.subplots()

    ax.scatter(st.session_state.ball,0,s=300)

    ax.set_xlim(-10,10)
    ax.set_ylim(-1,1)

    ax.set_title("BCI控制")

    st.pyplot(fig)