import os
import importlib
from typing import Dict, Type
from .base import BaseAlgorithm

class AlgorithmRegistry:
    _algorithms: Dict[str, Type[BaseAlgorithm]] = {}

    @classmethod
    def register(cls, algo_class):
        name = algo_class().name
        if name in cls._algorithms:
            raise ValueError(f"算法 {name} 已注册")
        cls._algorithms[name] = algo_class

    @classmethod
    def get(cls, name):
        if name not in cls._algorithms:
            raise ValueError(f"算法 {name} 不存在，可用：{list(cls._algorithms.keys())}")
        return cls._algorithms[name]

    @classmethod
    def discover(cls, path=None, package=None):
        if path is None:
            path = os.path.dirname(__file__)
        if package is None:
            package = __package__

        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(".py") and not f.startswith(("__", "base", "registry")):
                    mod_path = os.path.join(root, f)
                    mod_rel = os.path.relpath(mod_path, path)
                    mod_pkg = mod_rel.replace(os.sep, ".")[:-3]

                    try:
                        importlib.import_module(f".{mod_pkg}", package)
                    except Exception as e:
                        print(f"加载算法失败：{mod_pkg}，错误：{e}")

    @classmethod
    def list_algorithms(cls):
        return list(cls._algorithms.keys())