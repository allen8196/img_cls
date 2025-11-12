# src/engine.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from typing import Dict, Any, List

def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: Optimizer, 
    device: torch.device
) -> float:
    """
    執行一個週期的訓練

    Args:
        model (nn.Module): 訓練模型
        dataloader (DataLoader): 訓練資料載入器
        criterion (nn.Module): 損失函數
        optimizer (Optimizer): 優化器
        device (torch.device): 裝置 (cuda/cpu)

    Returns:
        float: 此週期的平均訓練損失
    """
    model.train()  # 設定為訓練模式
    total_loss: float = 0.0
    
    # 使用 tqdm 顯示進度條
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in progress_bar:
        # 將資料移動到指定裝置
        images, labels = images.to(device), labels.to(device)
        
        # 1. 梯度歸零
        optimizer.zero_grad()
        
        # 2. 前向傳播
        outputs = model(images)
        
        # 3. 計算損失
        loss = criterion(outputs, labels)
        
        # 4. 反向傳播
        loss.backward()
        
        # 5. 更新權重
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新進度條顯示
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

@torch.no_grad() # 確保在此函式中不計算梯度
def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Dict[str, float]:
    """
    在驗證集或測試集上評估模型

    Args:
        model (nn.Module): 評估模型
        dataloader (DataLoader): 資料載入器
        criterion (nn.Module): 損失函數
        device (torch.device): 裝置 (cuda/cpu)

    Returns:
        Dict[str, float]: 包含 loss, accuracy, precision, recall, f1, auc 的字典
    """
    model.eval()  # 設定為評估模式
    total_loss: float = 0.0
    
    # 用於儲存所有預測和標籤
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        
        # 計算機率 (Softmax) 和預測類別 (Argmax)
        # outputs shape: (batch_size, 2)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
        # 儲存 "PNEUMONIA" (類別 1) 的機率
        all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    
    # 轉換為 numpy array 以便 sklearn 計算
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # 計算各項指標
    accuracy = accuracy_score(y_true, y_pred)
    
    # 使用 'binary' average，因為我們是二分類 (PNEUMONIA vs NORMAL)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    try:
        # 計算 ROC AUC
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # 如果驗證集中只有一個類別 (例如 batch_size=1)，AUC 無法計算
        auc = 0.5 

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }
    
    return metrics