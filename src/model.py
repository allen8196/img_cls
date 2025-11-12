# src/model.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from typing import List

# 匯入設定
from src.config import config

def create_lora_efficientnetv2(
    num_classes: int = config.NUM_CLASSES, 
    lora_rank: int = config.LORA_RANK, 
    lora_alpha: int = config.LORA_ALPHA
) -> PeftModel:
    """
    創建一個應用了 LoRA 的 EfficientNetV2-S 模型。

    Args:
        num_classes (int): 輸出的類別數量
        lora_rank (int): LoRA 的 Rank (r)
        lora_alpha (int): LoRA 的 Alpha

    Returns:
        PeftModel: 應用 LoRA 之後的 PEFT 模型
    """
    
    # 1. 載入預訓練的 EfficientNetV2-S 模型
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)
    
    # 2. 替換最後的分類器層
    in_features = model.classifier[1].in_features
    # EfficientNetV2-S 的分類器是 (Dropout, Linear)
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # 3. 程式化地找到所有 Conv2d 層作為 LoRA 的目標模組
    target_modules: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 關鍵檢查：PEFT 要求 LoRA rank 必須能被 Conv2d 的 groups 整除
            # (rank % groups == 0)
            if lora_rank % module.groups == 0:
                target_modules.append(name)
            
    print(f"找到 {len(target_modules)} 個 Conv2d 層作為 LoRA 目標。")

    # 4. 定義 LoraConfig
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.LORA_DROPOUT,
        bias="none", # "none" or "all"
        # 關鍵：確保新加的分類器頭 (classifier) 被完整訓練 (而非 LoRA)
        modules_to_save=["classifier"]
    )

    # 5. 使用 PEFT 包裝模型
    lora_model = get_peft_model(model, lora_config)
    
    # 6. 打印可訓練參數的比例
    print("--- 模型參數狀態 (LoRA) ---")
    lora_model.print_trainable_parameters()
    print("----------------------------")
    
    return lora_model