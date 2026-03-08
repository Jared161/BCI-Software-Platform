from src.data_mgmt.query import BCIDataSystem

bci = BCIDataSystem(data_dir="./third_party_device_data")

print("可用数据：")
print(bci.query_data())

X, y, meta = bci.load_feature("exp_001")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("meta:", meta)