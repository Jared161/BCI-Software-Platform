import os
import importlib
from typing import Dict, Type
from .base import BaseAlgorithm

class AlgorithmRegistry:
    _algorithms: Dict[str, Type[BaseAlgorithm]] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, algo_class):
        name = algo_class().name
        if name in cls._algorithms:
            raise ValueError(f"算法 {name} 已注册")
        cls._algorithms[name] = algo_class

    @classmethod
    def get(cls, name):
        # 懒加载发现插件，避免调用方忘记显式 discover() 导致空注册表。
        if not cls._algorithms and not cls._discovered:
            cls.discover()

        if name not in cls._algorithms:
            raise ValueError(f"算法 {name} 不存在，可用：{list(cls._algorithms.keys())}")
        return cls._algorithms[name]

    @classmethod
    def discover(cls, path=None, package=None):
        # 默认发现流程只执行一次；传入自定义 path/package 时允许重复执行。
        if path is None and package is None and cls._discovered:
            return

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
                        if isinstance(e, ModuleNotFoundError):
                            missing_dep = getattr(e, "name", str(e))
                            print(f"加载算法失败：{mod_pkg}，缺少依赖：{missing_dep}。请先安装对应依赖后重试。")
                        else:
                            print(f"加载算法失败：{mod_pkg}，错误：{e}")

        if path == os.path.dirname(__file__) and package == __package__:
            cls._discovered = True

    @classmethod
    def list_algorithms(cls):
        if not cls._algorithms and not cls._discovered:
            cls.discover()
        return list(cls._algorithms.keys())