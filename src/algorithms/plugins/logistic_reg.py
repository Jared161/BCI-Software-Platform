import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from ..base import BaseAlgorithm
from ..__init__ import register_algorithm

@register_algorithm
class LogisticRegressionAlgorithm(BaseAlgorithm):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.learning_rate = self.params.get('learning_rate', 0.01)
        self.max_iter = self.params.get('max_iter', 1000)
        self.tolerance = self.params.get('tolerance', 1e-6)
        self.weights = None
        self.bias = None
        self.feature_names = None

    @property
    def name(self):
        return "logistic_reg"

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def set_params(self, **params) -> 'LogisticRegressionAlgorithm':
        self.params.update(params)
        self.learning_rate = self.params.get('learning_rate', 0.01)
        self.max_iter = self.params.get('max_iter', 1000)
        self.tolerance = self.params.get('tolerance', 1e-6)
        return self

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for i in range(self.max_iter):
            linear = X @ self.weights + self.bias
            y_pred = self._sigmoid(linear)
            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if np.linalg.norm(dw) < self.tolerance:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("模型未训练，请先调用 train()")
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("模型未训练，请先调用 train()")
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return np.hstack([1 - y_pred, y_pred])

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        print("\n" + "=" * 50)
        print("           逻辑回归算法评估结果")
        print("=" * 50)
        print(f"准确率      : {accuracy_score(y_test, y_pred):.4f}")
        print(f"精确率      : {precision_score(y_test, y_pred, zero_division=0,average='weighted'):.4f}")
        print(f"召回率      : {recall_score(y_test, y_pred, zero_division=0,average='weighted'):.4f}")
        print(f"F1分数      : {f1_score(y_test, y_pred, zero_division=0,average='weighted'):.4f}")
       # print(f"AUC         : {roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted'):.4f}")
       # 算法的逻辑回归是二分类算法，数据是四分类，跑不通，暂时注释掉
        print("\n混淆矩阵：")
        print(confusion_matrix(y_test, y_pred))
        print("=" * 50 + "\n")

def create_algorithm(params=None):
    return LogisticRegressionAlgorithm(params)