import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.algorithms.registry import AlgorithmRegistry

def main():
    parser = argparse.ArgumentParser(description="算法插件化运行框架")
    parser.add_argument("--algo", required=True, help="要运行的算法名称")
    parser.add_argument("--data", required=True, help="CSV数据集路径")
    parser.add_argument("--target", default="target", help="标签列名")
    args = parser.parse_args()

    # ====================== 自动发现所有算法 ======================
    AlgorithmRegistry.discover()

    # ====================== 自动实例化算法 ======================
    algo_class = AlgorithmRegistry.get(args.algo)
    algo = algo_class()

    # ====================== 自动加载你的CSV数据 ======================
    df = pd.read_csv(args.data)
    X = df.drop(args.target, axis=1).values
    y = df[args.target].values

    # ====================== 自动划分训练/测试集 ======================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ====================== 自动训练 ======================
    print(f"\n▶ 开始训练算法：{algo.name}")
    algo.train(X_train, y_train)

    # ====================== 自动评估 & 输出指标 ======================
    # 关键：所有输出都在算法内部，主程序 1 行调用即可
    algo.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()