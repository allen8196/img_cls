# src/config.py
"""
集中管理專案的超參數與設定
"""
import torch
from typing import List

class Config:
    # --- 環境設定 (Environment) ---
    # 自動偵測 CUDA 或 CPU
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 資料路徑 (Paths) ---
    # 資料集根目錄
    DATA_DIR: str = './data/chest_xray'
    # 訓練輸出目錄 (儲存模型權重)
    OUTPUT_DIR: str = './outputs'

    # --- 資料處理 (Data Processing) ---
    # 影像標準化的均值 (ImageNet)
    MEAN: List[float] = [0.485, 0.456, 0.406]
    # 影像標準化的標準差 (ImageNet)
    STD: List[float] = [0.229, 0.224, 0.225]
    
    # 影像尺寸。
    # 注意：EfficientNetV2-S 預訓練尺寸為 384x384，使用 384 效果優於 224
    IMAGE_SIZE: int = 384 

    # --- 資料集切分 (Splitting) ---
    # 以病患為單位切分，防止資料洩漏
    # 80% 訓練, 10% 驗證, 10% 測試
    TEST_SPLIT_SIZE: float = 0.2  # (驗證集 + 測試集) 的總比例
    VAL_SPLIT_SIZE: float = 0.5   # 從 TEST_SPLIT_SIZE 中，分配給驗證集的比例 (0.5 * 0.2 = 0.1)

    # --- 訓練超參數 (Training Hyperparameters) ---
    
    # 批次大小。
    # 注意：384x384 尺寸下，16GB VRAM 建議使用 16 或 12，32 可能導致 OOM
    BATCH_SIZE: int = 16
    
    # 訓練週期
    EPOCHS: int = 10
    
    # 初始學習率 (使用 AdamW，1e-4 是 LoRA fine-tuning 的良好起始點)
    LEARNING_RATE: float = 1e-4
    
    # AdamW 優化器的權重衰減
    WEIGHT_DECAY: float = 1e-2
    
    # DataLoader 的工作進程數
    NUM_WORKERS: int = 4
    
    # 是否啟用 pin_memory，加速 CPU 到 GPU 的資料傳輸
    PIN_MEMORY: bool = True
    
    # 學習率調度器 (ReduceLROnPlateau) 的耐心值
    # 當 F1-score 連續 2 個 epoch 沒有提升時，降低學習率
    SCHEDULER_PATIENCE: int = 2
    
    # Early Stopping
    # 當 val_f1_score 連續 4 個 epoch 沒有提升時，提前停止訓練
    # (此值應大於 SCHEDULER_PATIENCE)
    EARLY_STOPPING_PATIENCE: int = 4

    # --- 模型設定 (Model) ---
    # 類別數量 (0: NORMAL, 1: PNEUMONIA)
    NUM_CLASSES: int = 2
    
    # LoRA 參數: Rank (r)
    LORA_RANK: int = 8
    
    # LoRA 參數: Alpha
    LORA_ALPHA: int = 16
    
    # LoRA 參數: Dropout
    LORA_DROPOUT: float = 0.05

    # --- 損失函數 (Loss Function) ---
    # Focal Loss 參數 (用以處理類別不平衡)
    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0

# 實例化配置
config = Config()