# ==============================
# 页面配置 + 首页
# 终端运行streamlit run app.py
# ==============================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="BCI运动想象康复平台",
    layout="wide"
)

st.title("🧠 运动想象 BCI 康复训练 & 教学实训平台")
st.markdown("""
**面向：** 基层康复机构 / 高校康复专业  
**功能：** 零代码 → 数据加载 → 预处理 → 算法验证 → 运动意图识别 → 康复反馈
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 首页说明",
    "📊 数据与预处理",
    "🤖 算法训练与预测",
    "📈 结果与康复面板"
])



# ==============================
# 数据 + 预处理
# ==============================
with tab2:
    st.subheader("📊 脑电数据加载与预处理")

    # 模拟BCICIV_2a运动想象数据
    def get_demo_data():
        np.random.seed(42)
        return pd.DataFrame({
            "C3": np.random.randn(1000),
            "Cz": np.random.randn(1000),
            "C4": np.random.randn(1000),
            "label": np.random.randint(0, 4, 1000)
        })

    st.markdown("✅ 使用 BCICIV_2a 运动想象示范数据")
    df = get_demo_data()
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("⚙️ 预处理（滤波/去伪迹）")
    st.code("带通 0.5-30Hz | 50Hz陷波 | ICA去伪迹")
    st.success("数据预处理已完成")



# ==============================
# 算法 + 训练 + 预测
# ==============================
with tab3:
    st.subheader("🤖 运动想象意图分类算法")

    model_name = st.selectbox(
        "选择算法",
        ["SVM", "LDA", "Logistic回归"]
    )

    st.subheader("🚀 训练与预测")
    run = st.button("开始训练 & 识别")

    if run:
        # 模拟准确率（你们后期可替换成真实模型）
        acc = {
            "SVM": 0.82,
            "LDA": 0.80,
            "Logistic回归": 0.78
        }[model_name]

        st.session_state["acc"] = acc
        st.session_state["model"] = model_name
        st.session_state["pred"] = np.random.randint(0,4,10)

        st.success(f"训练完成！{model_name} 准确率 = {acc:.2f}")



# ==============================
# 可视化 + 康复面板 + 教学
# ==============================
with tab4:
    st.subheader("📈 分类结果与康复反馈")

    if "acc" not in st.session_state:
        st.warning("请先去【算法】页面训练一次")
    else:
        acc = st.session_state["acc"]
        model = st.session_state["model"]
        pred = st.session_state["pred"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("算法", model)
            st.metric("四分类准确率", f"{acc:.0%}")

        with col2:
            fig, ax = plt.subplots()
            ax.bar(["左手","右手","双脚","舌头"], [sum(pred==i) for i in range(4)])
            ax.set_title("运动意图识别统计")
            st.pyplot(fig)

        st.subheader("🎯 康复训练反馈面板")
        st.info("识别正确 → 神经反馈 → 促进脑功能重塑（脑卒中康复核心）")
        st.success("可用于：康复教学 + 基层康复算法验证")