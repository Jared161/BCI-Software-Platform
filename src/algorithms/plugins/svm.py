import numpy as np
from typing import Optional, Dict, Any
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from ..base import BaseAlgorithm
from ..__init__ import register_algorithm

@register_algorithm
class SVMAlgorithm(BaseAlgorithm):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.params = params or {}
        # 从params解析SVM参数（带默认值）
        self.kernel = self.params.get("kernel", "rbf")
        self.C = self.params.get("C", 1.0)
        self.gamma = self.params.get("gamma", "scale")
        self.random_state = self.params.get("random_state", 42)
        self.model = None
        self.feature_names = None

    @property
    def name(self):
        return "svm"

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def set_params(self, **params) -> 'SVMAlgorithm':
        self.params.update(params)
        self.kernel = self.params.get("kernel", "rbf")
        self.C = self.params.get("C", 1.0)
        self.gamma = self.params.get("gamma", "scale")
        self.random_state = self.params.get("random_state", 42)

        return self

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("SVM模型未训练！请先调用fit()方法训练")
        return self.model.predict(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        if self.model is None:
            raise ValueError("SVM模型未训练！请先调用train()方法训练")

        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0, average='weighted')
        recall = recall_score(y_test, y_pred, zero_division=0, average='weighted')
        f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')

        # 2. 完全复刻逻辑回归的打印格式（包括分隔线、中文标题、混淆矩阵）
        print("\n" + "=" * 50)
        print("           SVM算法评估结果")
        print("=" * 50)
        print(f"准确率      : {accuracy:.4f}")
        print(f"精确率      : {precision:.4f}")
        print(f"召回率      : {recall:.4f}")
        print(f"F1分数      : {f1:.4f}")
        # 可选：如果后续需要AUC，参考逻辑回归的注释风格保留
        # print(f"AUC         : {roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted'):.4f}")
        # SVM默认不输出概率，AUC需要开启probability=True，暂时注释
        print("\n混淆矩阵：")
        print(confusion_matrix(y_test, y_pred))
        print("=" * 50 + "\n")

        # 3. 保持原有返回字典的逻辑（不影响主程序调用）
        metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }
        return metrics