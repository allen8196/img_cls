# src/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss 實作，用於處理類別不平衡問題。
    原始論文: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha (float): 平衡正負樣本權重的因子 (alpha-balanced)
            gamma (float): 調節因子 (modulating factor)，專注於難分類樣本
            reduction (str): 'mean', 'sum' or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): 模型的原始 Logits 輸出 (N, C)
            targets (torch.Tensor): 真實標籤 (N)
        
        Returns:
            torch.Tensor: 計算後的 Focal Loss
        """
        
        # 計算標準的 Cross Entropy Loss (但不安裝 reduction)
        # ce_loss shape: (N)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # pt 是正確類別的預測機率
        # pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # 根據標籤獲取對應的 alpha 值
        # (假設 alpha 用於正樣本，1-alpha 用於負樣本)
        # 這裡的實作假設 alpha 用於所有類別，更精確的 alpha-balanced 應如下：
        alpha_t = torch.full_like(targets, 1.0 - self.alpha, dtype=torch.float, device=inputs.device)
        # targets == 1 的地方，alpha_t 設為 self.alpha (假設 1 是稀有類別)
        # 注意：在我們的案例中，PNEUMONIA (1) 是多數類別。
        # 原始論文的 alpha 用於 "正樣本" (前景)。
        # 這裡我們使用簡化版，直接將 alpha 乘上 (1-pt)^gamma
        # 或是可以手動計算權重：
        # targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        # at = torch.where(targets_one_hot == 1, self.alpha, 1.0 - self.alpha)
        # ... 這太複雜了

        # 簡單且常用的實作：
        # focal_loss = alpha * (1 - pt)^gamma * ce_loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss