import os           #找 GDF 文件、创建输出文件夹、拼接路径
import mne          #读取 BCICIV_2a 公开数据 的 gdf 文件，提取 EEG 信号、事件标签
import pandas as pd #把脑电数据变成表格，最后保存成.csv
import numpy as np  #处理脑电信号的多维数据（通道 × 时间点）



GDF_INPUT_DIR = r"D:\srtp大创\BCICIV_2a_gdf"  #原始GDF文件所在目录
CSV_OUTPUT_DIR = r"./third_party_device_data"   #转换后CSV输出目录
BCI2A_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P7', 'P5', 'P3', 'P1'
]      #BCICIV_2a 数据集标准22个EEG通道名称



def convert_gdf_to_csv(gdf_file_path: str, csv_save_path: str):   #转化目录下的一个gdf文件函数
    raw = mne.io.read_raw_gdf(gdf_file_path, preload=True, verbose=False) #读取GDF文件
    eeg_data = raw.get_data()  #提取EEG数据，通道×时间点
    events, event_id = mne.events_from_annotations(raw, verbose=False) #提取事件/标签，即运动想象标签
    sample_times = events[:, 0]
    labels = events[:, 2]   #事件格式：[时间点, 0, 标签]
    df = pd.DataFrame()    #构建DataFrame，标准CSV格式

    df['time'] = np.arange(eeg_data.shape[1]) / raw.info['sfreq']    #添加时间戳列（单位：秒）
    for idx, ch_name in enumerate(BCI2A_CHANNELS):
        df[ch_name] = eeg_data[idx]     #添加22个eeg通道

    df['label'] = 0
    for sample, label in zip(sample_times, labels):
        if sample < len(df):
            df.loc[sample, 'label'] = label

    df.to_csv(csv_save_path, index=False, encoding='utf-8')
    print("转化完成")



def batch_convert_all_gdf():     #批量转化目录下的gdf文件
    if not os.path.exists(GDF_INPUT_DIR):
        print(f"GDF输入目录 {GDF_INPUT_DIR}不存在")
        return

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)  #自动创建输出文件夹
    # 遍历所有.gdf文件
    gdf_files = [f for f in os.listdir(GDF_INPUT_DIR) if f.lower().endswith('.gdf')]
    if not gdf_files:
        print("未找到任何.gdf文件")
        return
    #遍历每一个 gdf 文件；生成对应的 csv 文件名；调用核心函数完成转换；自动保存到输出目录
    for gdf_file in gdf_files:
        gdf_path = os.path.join(GDF_INPUT_DIR, gdf_file)
        csv_filename = os.path.splitext(gdf_file)[0] + ".csv"
        csv_path = os.path.join(CSV_OUTPUT_DIR, csv_filename)
        convert_gdf_to_csv(gdf_path, csv_path)

if __name__ == "__main__":
    batch_convert_all_gdf()

