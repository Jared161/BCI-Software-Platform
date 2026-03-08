from src.algorithms.registry import AlgorithmRegistry

AlgorithmRegistry.discover()

print("可用算法：")
print(AlgorithmRegistry.list_algorithms())