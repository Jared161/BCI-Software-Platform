from src.data_mgmt.data_que_rea_int import load_feature
from src.data_mgmt.data_hie_dir_str import train_test_split
from src.algorithms.registry import AlgorithmRegistry
from sklearn.metrics import accuracy_score, f1_score


def run_pipeline(dataset_name, algo_name):

    # 1. 加载数据
    X, y, meta = load_feature(dataset_name)

    # 2. 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 3. 获取算法
    AlgoClass = AlgorithmRegistry.get(algo_name)
    model = AlgoClass()

    # 4. 训练
    model.train(X_train, y_train)

    # 5. 预测
    y_pred = model.predict(X_test)

    # 6. 计算指标
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    metrics = {
        "accuracy": acc,
        "f1": f1
    }

    print("Metrics:", metrics)

    return metrics