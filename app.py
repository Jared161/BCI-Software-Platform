import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# pipeline可视化组件
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState

# 在终端输入streamlit run app.py打开网页
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入原有模块
from src.pipeline.run_pipeline import run_pipeline
from src import AlgorithmRegistry
from src.preprocessing.band_pass_filter import BandpassFilter

# 页面配置
st.set_page_config(
    page_title="BCI脑机接口实验平台",
    page_icon="⭐",
    layout="wide"
)

# ---------------------- 标题 ----------------------
st.title("BCI脑机接口实验平台")
st.markdown("### 计算机设计大赛参赛作品")
st.divider()

# ---------------------- 页面导航 ----------------------
page = st.sidebar.radio(
    "平台功能",
    ["算法实验学习", "EEG实时可视化演示", "Pipeline编排", "教学案例学习"]
)

# ---------------------- 算法实验模块 ----------------------
if page == "算法实验学习":

    with st.sidebar:
        st.header("实验配置")

        run_mode = st.selectbox(
            "运行模式",
            ["单算法实验（带滤波）", "算法基准测试"],
            index=0
        )

        algo_list = AlgorithmRegistry.list_algorithms()

        selected_algo = st.selectbox("选择算法", algo_list) if run_mode == "单算法实验（带滤波）" else None

        st.subheader("预处理参数")

        lowcut = st.slider("带通滤波低频(Hz)", 1, 20, 8)
        highcut = st.slider("带通滤波高频(Hz)", 20, 50, 30)
        fs = st.slider("采样率(Hz)", 128, 500, 250)

    st.subheader("实验控制")

    if st.button("开始实验", type="primary"):

        with st.spinner("实验中..."):

            try:

                if run_mode == "单算法实验（带滤波）":

                    metrics = run_pipeline(algo_name=selected_algo)

                    st.success("实验完成！")

                    result_df = pd.DataFrame({
                        "评估指标": ["Accuracy", "F1 Score"],
                        "数值": [round(metrics["accuracy"],4), round(metrics["f1"],4)]
                    })

                    st.table(result_df)

                    fig, ax = plt.subplots()

                    ax.bar(result_df["评估指标"], result_df["数值"])

                    ax.set_ylim(0,1.1)

                    ax.set_title(f"{selected_algo.upper()} 实验结果")

                    st.pyplot(fig)

                    st.info(f"滤波参数 {lowcut}-{highcut}Hz  采样率 {fs}Hz")

                else:

                    from src.experiments import run_benchmark

                    run_benchmark()

                    st.success("基准测试完成")

            except Exception as e:

                st.error(str(e))


# ---------------------- EEG实时可视化 ----------------------
#暂时为模拟刺激，
elif page == "EEG实时可视化演示":

    st.header("实时EEG信号演示")

    st.write("模拟脑电信号实时显示")

    placeholder = st.empty()

    if st.button("开始采集"):

        for i in range(80):

            t = np.linspace(0,1,500)

            signal = np.sin(10*2*np.pi*t + i*0.1) + 0.3*np.random.randn(500)

            fig, ax = plt.subplots()

            ax.plot(signal)

            ax.set_title("EEG Signal")

            placeholder.pyplot(fig)

            time.sleep(0.1)

# ---------------------- Pipeline拖拽模块 ----------------------
elif page == "Pipeline编排":

    st.header("BCI Pipeline Builder")

    st.write("拖动节点构建脑机接口处理流程")

    nodes = [
        StreamlitFlowNode(
            id='1',
            pos=(100,100),
            data={'label':'Raw EEG'}
        ),
        StreamlitFlowNode(
            id='2',
            pos=(300,100),
            data={'label':'Bandpass Filter'}
        ),
        StreamlitFlowNode(
            id='3',
            pos=(500,100),
            data={'label':'Feature Extraction'}
        ),
        StreamlitFlowNode(
            id='4',
            pos=(700,100),
            data={'label':'Classifier'}
        ),
    ]

    edges = [
        StreamlitFlowEdge('1','2','3'),
        StreamlitFlowEdge('2','3','4'),
        StreamlitFlowEdge('3','4','5')
    ]

    if 'flow_state' not in st.session_state:

        st.session_state.flow_state = StreamlitFlowState(nodes,edges)

    st.session_state.flow_state = streamlit_flow(
        'bci_flow',
        st.session_state.flow_state
    )

    st.info("""
BCI典型流程：

Raw EEG  
↓  
Bandpass Filter  
↓  
Feature Extraction  
↓  
Classifier
""")

# ---------------------- 教学案例 ----------------------
elif page == "教学案例学习":

    st.header("BCI教学案例")

    case = st.selectbox(
        "选择案例",
        ["SSVEP演示","运动想象MI"]
    )

    if case == "SSVEP演示":

        st.subheader("SSVEP模拟")

        freq = st.slider("刺激频率",5,20,10)

        t = np.linspace(0,1,500)

        signal = np.sin(2*np.pi*freq*t)

        fig, ax = plt.subplots()

        ax.plot(signal)

        ax.set_title("SSVEP信号")

        st.pyplot(fig)

        st.info("SSVEP通过不同频率视觉刺激产生对应脑电响应")

    elif case == "运动想象MI":

        st.subheader("运动想象实验")

        direction = st.radio("想象运动",["左手","右手"])

        if direction == "左手":

            st.success("检测到左手运动想象")

        else:

            st.success("检测到右手运动想象")

# ---------------------- 底部 ----------------------
st.divider()

st.markdown("""
### 平台说明

功能模块：

- EEG实时信号可视化
- BCI算法实验
- 图形化Pipeline构建
- BCI教学案例演示

应用场景：

- BCI教学实验
- 脑机接口算法验证
- 可视化信号处理流程
""")