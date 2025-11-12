# src/utils.py
"""
通用工具函式，例如儲存和載入模型檢查點
"""
import torch
import torch.nn as nn
import os
from typing import Union

def save_checkpoint(model: nn.Module, filepath: str) -> None:
    """
    儲存模型檢查點 (state_dict)

    Args:
        model (nn.Module): 欲儲存的 PyTorch 模型
        filepath (str): 儲存路徑 (例如: ./outputs/best_model.pth)
    """
    try:
        # 確保儲存目錄存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # 儲存模型的狀態字典
        torch.save(model.state_dict(), filepath)
        print(f"✅ 模型成功儲存至: {filepath}")
    except Exception as e:
        print(f"❌ 儲存模型失敗: {e}")

def load_checkpoint(model: nn.Module, filepath: str, device: torch.device) -> None:
    """
    載入模型檢查點 (state_dict)

    Args:
        model (nn.Module): 欲載入權重的 PyTorch 模型 (需有相同架構)
        filepath (str): 權重檔案路徑
        device (torch.device): 欲將模型載入的裝置 (cpu 或 cuda)
    """
    try:
        if not os.path.exists(filepath):
            print(f"⚠️ 警告：找不到檢查點檔案: {filepath}。將使用預訓練權重。")
            return
            
        # 載入狀態字典
        state_dict = torch.load(filepath, map_location=device)
        # 載入權重至模型
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"✅ 模型成功從 {filepath} 載入。")
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")