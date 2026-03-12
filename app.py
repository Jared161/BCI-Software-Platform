import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 在终端输入streamlit run app.py打开网页
# 解决项目模块导入问题
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入核心模块
from src.pipeline.run_pipeline import run_pipeline
from src import AlgorithmRegistry
from src.preprocessing.band_pass_filter import BandpassFilter

# 设置页面样式
st.set_page_config(
    page_title="BCI脑机接口实验平台",
    page_icon="⭐",
    layout="wide"
)

# ---------------------- 页面标题与说明 ----------------------
st.title("BCI脑机接口实验平台（轻量版）")
st.markdown("### 北科大计算机设计大赛参赛作品")
st.divider()

# ---------------------- 侧边栏配置（算法/参数选择） ----------------------
with st.sidebar:
    st.header("实验配置")
    # 1. 选择运行模式
    run_mode = st.selectbox(
        "运行模式",
        ["单算法实验（带滤波）", "算法基准测试"],
        index=0
    )
    # 2. 选择算法（仅单算法模式显示）
    algo_list = AlgorithmRegistry.list_algorithms()  # 获取所有注册的算法（svm/logistic_reg/knn）
    selected_algo = st.selectbox("选择算法", algo_list) if run_mode == "单算法实验（带滤波）" else None
    # 3. 滤波参数配置
    st.subheader("预处理参数")
    lowcut = st.slider("带通滤波低频(Hz)", 1, 20, 8)  # 默认8Hz
    highcut = st.slider("带通滤波高频(Hz)", 20, 50, 30)  # 默认30Hz
    fs = st.slider("采样率(Hz)", 128, 500, 250)  # 默认250Hz

# ---------------------- 核心功能：一键运行实验 ----------------------
st.subheader("实验控制")
# 一键运行按钮
if st.button("开始实验", type="primary"):
    # 添加加载动画
    with st.spinner("实验中...（数据加载→带通滤波→模型训练→结果评估）"):
        try:
            if run_mode == "单算法实验（带滤波）":
                # 运行带滤波的pipeline，传入自定义参数（可选，这里用默认）
                metrics = run_pipeline(algo_name=selected_algo)

                # 展示实验结果（可视化）
                st.success("实验完成！")
                st.subheader("实验结果")
                # 1. 结果表格
                result_df = pd.DataFrame({
                    "评估指标": ["准确率(Accuracy)", "F1分数(F1-Score)"],
                    "数值": [round(metrics["accuracy"], 4), round(metrics["f1"], 4)]
                })
                st.table(result_df)

                # 2. 简单可视化（柱状图，更直观）
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(result_df["评估指标"], result_df["数值"], color=["#1f77b4", "#ff7f0e"])
                ax.set_ylim(0, 1.1)  # 指标范围0-1
                ax.set_ylabel("数值")
                ax.set_title(f"{selected_algo.upper()} 算法实验结果")
                st.pyplot(fig)

                # 3. 补充信息
                st.info(f"预处理参数：带通滤波 {lowcut}-{highcut}Hz，采样率 {fs}Hz")

            else:
                # 运行基准测试（批量对比所有算法）
                from src.experiments import run_benchmark

                run_benchmark()
                st.success("基准测试完成！所有算法结果已输出到控制台")

        # 异常处理
        except PermissionError as e:
            st.error(
                f"权限错误：{e}\n解决方案：手动创建目录 D:\\pycharm\\code\\BCI-Software-Platform\\third_party_device_data")
        except ValueError as e:
            st.error(f"数据错误：{e}\n解决方案：在 third_party_device_data/csv 下放exp_001.csv示例数据")
        except Exception as e:
            st.error(f"实验出错：{str(e)}")

# ---------------------- 底部说明（提升专业度） ----------------------
st.divider()
st.markdown("""
### ♥ 平台说明
- 核心功能：插件化算法框架 + EEG带通滤波预处理 + 一键式实验流程
- 支持算法：SVM / 逻辑回归 / KNN（可快速扩展）
- 应用场景：高校BCI脑机接口教学、低成本科研实验
""")