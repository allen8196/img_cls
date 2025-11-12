# src/main.py
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

# 匯入本地模組
from src.dataset import create_patient_level_split, ChestXRayDataset, get_transforms
from src.model import create_lora_efficientnetv2
from src.loss import FocalLoss
from src.engine import train_one_epoch, evaluate
from src.utils import save_checkpoint, load_checkpoint
from src.config import config

def main():
    # 使用 config 中的裝置設定
    device = config.DEVICE
    print(f"使用裝置: {device}")
    
    # 建立輸出目錄
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 初始化 W&B Run ---
    # 將 config 物件轉換為字典以便 wandb 記錄 (排除內建屬性)
    config_dict = {
        k: getattr(config, k) 
        for k in dir(config) 
        if not k.startswith('__') and not callable(getattr(config, k))
    }
    
    try:
        wandb.init(
            project="Pneumonia-Classification-EfficientNetV2", # 專案名稱
            config=config_dict, # 記錄所有超參數
            name=f"run_img{config.IMAGE_SIZE}_bs{config.BATCH_SIZE}_lr{config.LEARNING_RATE}_lora{config.LORA_RANK}" # 自訂 Run 名稱
        )
        print("✅ Weights & Biases 監控已啟動。")
    except Exception as e:
        print(f"❌ 無法初始化 W&B (請檢查 API 金鑰是否設定): {e}")
        wandb.init(mode="disabled") # 即使失敗也繼續執行 (禁用 W&B)


    # --- [SWEEP] 獲取 Sweeps 參數並覆蓋 config ---
    # wandb.config 會包含 default config 和 sweep 傳入的新參數
    # 我們用 wandb.config 的值來 *更新* 我們的 config 物件
    print("--- [SWEEP] 正在檢查並應用 Sweep 參數 ---")
    try:
        # [重要] 從 wandb.config 回寫到 config 物件
        config.LEARNING_RATE = wandb.config.LEARNING_RATE
        config.LORA_RANK = wandb.config.LORA_RANK
        config.LORA_ALPHA = wandb.config.LORA_ALPHA
        config.FOCAL_ALPHA = wandb.config.FOCAL_ALPHA
        
        # 更新 Run 名稱以反映 Sweep 參數
        sweep_name = f"sweep_lr{config.LEARNING_RATE:.0e}_rank{config.LORA_RANK}_alpha{config.LORA_ALPHA:.2f}"
        wandb.run.name = sweep_name
        
        print(f"  [SWEEP] 成功應用 Sweep 參數。")
        print(f"  [SWEEP] 學習率: {config.LEARNING_RATE}")
        print(f"  [SWEEP] LoRA Rank: {config.LORA_RANK}")
        print(f"  [SWEEP] LoRA Alpha: {config.LORA_ALPHA}")
        print(f"  [SWEEP] Focal Alpha: {config.FOCAL_ALPHA}")
    except AttributeError as e:
        print(f"  [SWEEP] 未執行 Sweep (或參數名稱不符)，使用 config.py 預設值。")
    # ------------------------------------------------


    # 1. 資料準備
    print("正在準備資料集...")
    # 接收完整的 train, val, test 切分
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = \
        create_patient_level_split(
            config.DATA_DIR, 
            test_size=config.TEST_SPLIT_SIZE,
            val_size_ratio=config.VAL_SPLIT_SIZE
        )
    
    # 獲取影像轉換
    train_transform, val_transform = get_transforms(image_size=config.IMAGE_SIZE)
    
    # 建立 Dataset
    train_dataset = ChestXRayDataset(train_files, train_labels, transform=train_transform)
    val_dataset = ChestXRayDataset(val_files, val_labels, transform=val_transform)
    
    # 建立 Test Dataset
    test_dataset = ChestXRayDataset(test_files, test_labels, transform=val_transform)
    
    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    
    # 建立 Test DataLoader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"訓練集: {len(train_dataset)} 樣本, {len(train_loader)} 批次")
    print(f"驗證集: {len(val_dataset)} 樣本, {len(val_loader)} 批次")
    print(f"測試集: {len(test_dataset)} 樣本, {len(test_loader)} 批次")

    # 2. 模型、損失函數、優化器
    print("正在建立模型...")
    # 移除多餘的回傳值
    model = create_lora_efficientnetv2(
        num_classes=config.NUM_CLASSES, 
        lora_rank=config.LORA_RANK, 
        lora_alpha=config.LORA_ALPHA
    )
    model.to(device)
    
    # 損失函數
    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    
    # 優化器 (僅優化可訓練的參數)
    optimizer = AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 學習率調度器 (監控驗證集的 F1-score)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', # 'max' 因為我們監控 F1-score
        factor=0.1, 
        patience=config.SCHEDULER_PATIENCE,
    )
    
    # 監控模型梯度與參數 ---
    wandb.watch(model, criterion, log="all", log_freq=100) # 每 100 批次記錄一次

    # 3. 訓練迴圈
    best_f1: float = 0.0
    best_model_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
    # 初始化 early_stopping 計數器 
    early_stopping_counter: int = 0
    print(f"--- 開始訓練，共 {config.EPOCHS} 個 Epoch ---")

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        
        # 訓練
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} 訓練損失: {train_loss:.4f}")
        
        # 驗證
        metrics = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} 驗證指標:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # 記錄指標
        # 準備要 log 的資料字典
        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": metrics['loss'],
            "val_accuracy": metrics['accuracy'],
            "val_f1_score": metrics['f1_score'],
            "val_auc": metrics['auc'],
            "val_precision": metrics['precision'],
            "val_recall": metrics['recall'],
            "learning_rate": optimizer.param_groups[0]['lr'] # 記錄當前學習率
        }
        wandb.log(log_data)
            
        # 儲存與檢查邏輯
        current_f1 = metrics['f1_score']
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            print(f"🚀 新高 F1-score: {best_f1:.4f}。儲存模型至 {best_model_path}...")
            save_checkpoint(model, best_model_path)
            # 儲存最佳 F1-score 到 summary
            wandb.run.summary["best_val_f1_score"] = best_f1
            
            # 重置 early_stopping 計數器
            early_stopping_counter = 0 
        else:
            # 未見改善，計數器+1
            early_stopping_counter += 1
            print(f"Epoch {epoch+1} 未見改善. Early Stopping 計數: {early_stopping_counter}/{config.EARLY_STOPPING_PATIENCE}")

        # 更新學習率 (在 F1 檢查之後)
        scheduler.step(current_f1)
        
        # 檢查是否觸發 early_stopping
        if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"--- 觸發 Early Stopping (Patience={config.EARLY_STOPPING_PATIENCE}) ---")
            # [WANDB] 記錄停止的 Epoch
            wandb.run.summary["stopped_epoch"] = epoch + 1
            wandb.log({"early_stopped": True})
            break # 跳出 epoch 迴圈

    # 4. 最終測試
    print("\n--- 訓練完成 ---")
    print(f"載入最佳模型 (F1: {best_f1:.4f}) 進行最終測試...")
    
    # 建立一個新的模型實例並載入最佳權重
    # (確保我們使用的是最佳模型，而不僅是最後一個 Epoch 的模型)
    final_model = create_lora_efficientnetv2(
        num_classes=config.NUM_CLASSES, 
        lora_rank=config.LORA_RANK, 
        lora_alpha=config.LORA_ALPHA
    )
    load_checkpoint(final_model, best_model_path, device)
    
    # 在測試集上評估
    test_metrics = evaluate(final_model, test_loader, criterion, device)
    
    print("\n--- 最終測試指標 (Test Set) ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 記錄最終測試結果
    test_log_data = {f"test_{k}": v for k, v in test_metrics.items()}
    wandb.log(test_log_data)
    # 將測試指標儲存到 Run 的 Summary 中
    for k, v in test_metrics.items():
        wandb.run.summary[f"final_test_{k}"] = v

    wandb.finish()

    print("\n專案執行完畢。")

if __name__ == '__main__':
    main()