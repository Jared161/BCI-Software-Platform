import numpy as np
from typing import Optional, Dict, Any
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
            raise ValueError("SVM模型未训练！请先调用fit()方法训练")

        y_pred = self.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted')
        }
        return metrics