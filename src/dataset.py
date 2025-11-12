# src/dataset.py
import os
import re
from glob import glob
from collections import defaultdict
import random
from typing import List, Tuple, Dict, Any, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# 匯入設定
from src.config import config

SplitDataType = Tuple[List[str], List[int]]

def create_patient_level_split(
    data_dir: str, 
    test_size: float = 0.2, 
    val_size_ratio: float = 0.5
) -> Tuple[SplitDataType, SplitDataType, SplitDataType]:
    """
    以病患為單位切分資料集，防止同病患的影像洩漏至不同資料集。
    
    檔名格式假設為 (類別)-(病患ID)-(影像編號).jpeg
    例如: PNEUMONIA-person100_bacteria_480.jpeg
         NORMAL-NORMAL2-IM-0001-0001.jpeg

    Args:
        data_dir (str): 資料集根目錄 (包含 train/test/val 子目錄)
        test_size (float): 驗證集+測試集的總比例 (例如 0.2)
        val_size_ratio (float): 從 test_size 中，分配給驗證集的比例 (例如 0.5，表示 0.2*0.5=0.1 給驗證集)

    Returns:
        Tuple[SplitDataType, SplitDataType, SplitDataType]: 
            (train_files, train_labels), 
            (val_files, val_labels), 
            (test_files, test_labels)
    """
    
    # 遞迴獲取所有 .jpeg 影像路徑
    all_image_paths = glob(os.path.join(data_dir, '**', '*.jpeg'), recursive=True)
    
    # [修正] 修正 defaultdict 的 lambda 語法
    patient_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'paths': [], 'label': None})
    
    print(f"正在分析 {len(all_image_paths)} 張影像...")

    # 根據病患ID組織資料
    for path in all_image_paths:
        try:
            # 從路徑中獲取類別 (NORMAL or PNEUMONIA)
            label_str = os.path.basename(os.path.dirname(path))
            label = 0 if label_str == 'NORMAL' else 1
            
            # 從檔名中解析病患ID
            filename = os.path.basename(path)
            
            # 嘗試多種正則表達式來匹配病患ID
            match = re.search(r'person(\d+)_', filename)           # 格式: person100_...
            if not match:
                match = re.search(r'NORMAL2-IM-(\d+)-', filename) # 格式: NORMAL2-IM-0001-...
            if not match:
                match = re.search(r'IM-(\d+)-', filename)         # 格式: IM-0001-...
            
            if match:
                # 為了確保不同類別的相同ID不被混淆，我們將類別加入ID
                patient_id = f"{label_str}_{match.group(1)}"
            else: 
                # 如果格式不符，則將去除副檔名的檔名本身作為唯一ID (最壞情況)
                patient_id = filename.split('.')[0]

            patient_data[patient_id]['paths'].append(path)
            patient_data[patient_id]['label'] = label
            
        except Exception as e:
            print(f"解析路徑 {path} 失敗: {e}")

    patient_ids = list(patient_data.keys())
    random.seed(42) # 確保可重現
    random.shuffle(patient_ids)
    
    print(f"共找到 {len(patient_ids)} 位獨立病患。")

    # 切分訓練集和臨時集 (驗證集+測試集)
    train_pids, temp_pids = train_test_split(
        patient_ids, 
        test_size=test_size, 
        random_state=42
    )
    
    # 從臨時集中切分驗證集和測試集
    val_pids, test_pids = train_test_split(
        temp_pids, 
        test_size=val_size_ratio, # 此處 test_size 意指 val_size_ratio
        random_state=42
    )

    def get_files_from_pids(pids: List[str]) -> SplitDataType:
        files: List[str] = []
        labels: List[int] = []
        for pid in pids:
            files.extend(patient_data[pid]['paths'])
            labels.extend([patient_data[pid]['label']] * len(patient_data[pid]['paths']))
        return files, labels

    train_files, train_labels = get_files_from_pids(train_pids)
    val_files, val_labels = get_files_from_pids(val_pids)
    test_files, test_labels = get_files_from_pids(test_pids)
    
    print(f"資料切分完成:")
    print(f"  訓練集: {len(train_files)} 張影像 ({len(train_pids)} 位病患)")
    print(f"  驗證集: {len(val_files)} 張影像 ({len(val_pids)} 位病患)")
    print(f"  測試集: {len(test_files)} 張影像 ({len(test_pids)} 位病患)")
    
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


class ChestXRayDataset(Dataset):
    """
    胸部 X 光影像的 PyTorch Dataset
    """
    def __init__(self, file_paths: List[str], labels: List[int], transform: Callable = None):
        """
        Args:
            file_paths (List[str]): 影像檔案路徑列表
            labels (List[int]): 對應的標籤列表 (0 或 1)
            transform (Callable, optional): 應用於影像的轉換 (Augmentation)
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.file_paths[idx]
        
        # 載入影像並轉換為 'RGB'
        # (預訓練模型需要 3 通道，即使 X 光是灰階)
        image = Image.open(img_path).convert('RGB')
        
        # 標籤轉換為 Tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(
    image_size: int = config.IMAGE_SIZE, 
    mean: List[float] = config.MEAN, 
    std: List[float] = config.STD
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    獲取訓練集和驗證/測試集的影像轉換流程
    """
    
    # 訓練集的資料增強 (Data Augmentation)
    train_transform = transforms.Compose([
        # 隨機裁切並縮放到指定大小
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        # 隨機水平翻轉
        transforms.RandomHorizontalFlip(p=0.5),
        # 隨機旋轉 (-15 到 +15 度)
        transforms.RandomRotation(degrees=15),
        # 轉換為 Tensor
        transforms.ToTensor(),
        # 標準化 (使用 ImageNet 統計數據)
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 驗證集與測試集的轉換 (不含隨機性)
    val_test_transform = transforms.Compose([
        # 縮放 (保持長寬比，短邊縮放至 image_size * 256/224)
        transforms.Resize(int(image_size * 256 / 224)),
        # 中心裁切
        transforms.CenterCrop(image_size),
        # 轉換為 Tensor
        transforms.ToTensor(),
        # 標準化
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_test_transform