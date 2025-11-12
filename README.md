# X 光影像分類（EfficientNetV2-S + LoRA）

本專案使用 **LoRA** 對 **EfficientNetV2-S** 模型進行微調，以解決 [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 資料集的二元分類任務。

此流程整合資料切分策略、處理類別不平衡的損失函數，並利用 Weights & Biases 進行訓練追蹤與超參數優化（Sweeps）。

## 核心技術

* **模型選用**：`EfficientNetV2-S`（使用 `torchvision` 載入 ImageNet 預訓練權重）
* **微調策略**：PEFT（LoRA）（`peft` 函式庫）
* **資料處理**：以病患 ID 為單位重新切分資料集 (80/10/10)，防止同一病患影像跨足訓練/驗證集。
* **影像尺寸**：使用 `384x384`，匹配 `EfficientNetV2-S` 的預訓練規格，以提高特徵提取效果。
* **損失函數**：自定義 `Focal Loss`，用以處理資料類別不平衡。
* **訓練策略**：AdamW 優化器、`ReduceLROnPlateau` 自動調整學習率、`Early Stopping` 防止過擬合。
* **實驗追蹤**：Weights & Biases（Wandb）
* **超參數優化**：使用 W&B Sweeps 進行貝氏優化，自動調整超參數。

---

## 專案架構

```
/pneumonia-classification
 |-- data/              # （需手動下載）Kaggle 資料集
 |-- src/
 | |-- config.py        # 集中管理所有超參數與路徑
 | |-- dataset.py       # 資料載入、病患級別切分、資料增強
 | |-- model.py         # 模型定義、LoRA 配置
 | |-- engine.py        # 訓練與驗證迴圈
 | |-- loss.py          # 定義損失函數 Focal Loss
 | |-- utils.py         # 工具函式（儲存/載入模型檢查點）
 | |-- main.py          # 主入口：整合所有模組、執行訓練流程
 |-- sweep.yaml         # Wandb Sweeps 超參數優化設定檔
 |-- requirements.txt 
 ```

## 系統需求與環境設置

本專案基於 Python 3.12.4、CUDA 13.0 的環境執行。

### 1. 環境建立

以 `uv` 為例進行環境管理。

```bash
# 安裝 uv (若尚未安裝)
pip install uv
# 指定 Python 版本
pyenv local 3.12.4
# 建立虛擬環境
uv venv
# 啟動虛擬環境
source .venv/bin/activate
# 安裝依賴
uv pip install -r requirements.txt
```

### 2. 資料準備

1.  從 Kaggle 下載 [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 資料集。
2.  解壓縮後，將 `chest_xray` 資料夾（包含 `train`, `test`, `val` 子目錄）完整移動到 `data/` 資料夾。

### 3. （可選）登入 W&B

推薦使用 Weights & Biases 追蹤訓練狀況及優化超參數。需先註冊帳號並取得 API Key。
若不登入，腳本將自動切換至「禁用」模式。

```bash
# 登入 W&B (會提示貼上 API Key)
wandb login
```

### 4. 執行訓練
提供兩種訓練模式：

1. 單次訓練（使用預設參數）
此方式使用 `src/config.py` 中定義的預設超參數，進行一次完整的訓練。
```bash
python -m src.main
```

2. 超參數優化（W&B Sweep）
此方式使用 sweep.yaml 中的設定，使用 W&B 進行貝氏優化，自動尋找最佳超參數組合。
```bash
# 1. 初始化 Sweep
# W&B 伺服器將建立一個 Sweep 任務，並回傳 SWEEP_ID
wandb sweep sweep.yaml
# 2. 啟動 Agent
# 將 <YOUR_SWEEP_ID> 替換為上一步回傳的 ID
# Agent 會不斷向 W&B 伺服器詢問下一組參數並執行
wandb agent <YOUR_SWEEP_ID>
```
---

## 實驗結果與分析

本專案使用 W&B Sweeps 進行了 4 次貝氏優化搜索。

### 1. Sweep 效能總覽

以下為 W&B Sweep 運行的最終測試集 (Test Set) 效能摘要：

| N. | Learning Rate | LoRA Rank | LoRA Alpha | val_f1_score | **test_f1_score** | **test_auc** | **test_accuracy** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1.01e-4 | 8 | 32 | 98.72% | **98.17%** | 99.15% | 97.30% |
| 2 | 1.87e-4 | 16 | 64 | 98.84% | 98.06% | 99.42% | 97.13% |
| 3 | 1.78e-4 | 16 | 32 | 98.73% | 97.83% | 99.41% | 96.80% |
| 4 | 5.32e-5 | 16 | 16 | 98.95% | 97.71% | 99.10% | 96.63% |


### 2. 結果分析

* **高效能**：所有 Sweep 組合在測試集上均達到 `F1-Score > 97.5%` 且 `AUC > 99%`。模型泛化能力出色。
* **最佳參數組合**：`Rank=8`, `Alpha=32` 的組合在測試集上取得了最佳的 F1-Score。故 Rank = 8 在此任務上已足夠，參數效率高。
* **訓練策略驗證**：四次訓練中，有兩次在第 8-9 Epoch 觸發了提前停止，有效防止過擬合並節省了訓練資源。

### 3. 系統資源基準
基於 NVIDIA RTX 5070 Ti 16G，訓練時的資源用量：
* **VRAM 佔用**：在 `IMAGE_SIZE=384`、`BATCH_SIZE=16` 的設定下，訓練期間 VRAM 使用率約在 **~13 GB**，無 OOM 風險。
（若 VRAM < 16GB，應降低 `BATCH_SIZE`，或使用梯度累積。）
* **GPU 使用率**：訓練期間 GPU 使用率維持在 80~90%，表示 GPU 得到充分利用。
* **CPU / RAM**：訓練過程為 GPU-bound，CPU 使用率低（約 15~20%）、RAM 佔用低（約 2.2 GB）。

### 4. Baseline 對比：LoRA vs. Linear Probing

為了驗證 LoRA 微調的必要性，本專案以 Linear Probing 為 Baseline 進行對比。

* **LoRA**：訓練 `classifier`（分類頭）+ `Conv2d` 的 LoRA 適配器。
* **Linear Probing**：僅訓練 `classifier`（分類頭），凍結其他權重。

**對比結果 (Test Set)**：

| 微調策略 | test_f1_score | test_auc | test_accuracy | test_precision | test_recall | test_loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Linear Probing | 95.11% | 96.25% | 92.75% | 93.51% | 96.76% | 0.014105 |
| **LoRA** | **98.18%** | **99.42%** | **97.58%** | **96.64%** | **99.77%** | **0.007219** |

**分析**：

* **LoRA 效能勝出**：LoRA 策略在各項指標均優於 Baseline，其中 `test_f1_score` 提升了 **+3.07%**。
* **領域適應的必要性**：此對比證實，`EfficientNetV2-S` 效能出色，在 ImageNet 的通用預訓練特徵已可直接處理專業醫學影像任務，但微調仍可明顯提升效能。
![Linear Probing 與 LoRA 對比](https://upload.cc/i1/2025/11/12/E0YxnR.png)
---

## 運作邏輯與設計思路

### 1. 資料處理 (`src/dataset.py`)

#### 防止資料洩漏

問題：Kaggle 上的原始資料集**以影像檔為單位**切分。這可能導致同一位病患的 X 光片同時出現在訓練集和驗證集中，使模型學習到「病患特徵」而非「肺炎病徵」，導致驗證集準確率虛高。

解決：依據檔名以「病患 ID」為單位，重新切分為 80% 訓練、10% 驗證、10% 測試集。

#### 資料擴增 (Data Augmentation)

* 訓練集：隨機裁切並縮放至 384x384、水平翻轉、旋轉 (±15°)。
* 驗證/測試集：僅調整尺寸至 384x384。


### 2. 模型選用與 LoRA 微調 (`src/model.py`)

#### 模型選用：EfficientNetV2-S

* 相較於 ResNet，EfficientNetV2 參數效率高，適合訓練資料少、訓練硬體有限的情況。
* EfficientNetV2 的預訓練資料尺寸為 384x384（ResNet 為 224x224），有助於捕捉更細微的特徵。

#### 微調策略：LoRA

* 由於訓練資料較少（約 5k 張圖片），為避免過擬合，採用 LoRA 對部分 `Conv2d` 層進行 LoRA 微調。
* 為適應新的二元分類任務，同時對新替換的分類頭進行全參數微調。

### 3. 損失函數與訓練策略 (`src/loss.py` & `src/engine.py`)

#### 損失函數：Focal Loss 

* 此資料集存在類別不平衡（肺炎樣本遠多於正常樣本），易導致標準 Cross Entropy 會被多數類別主導。（模型會傾向於預測有肺炎）
* Focal Loss 改良自 Cross Entropy，藉由調節因子，降低「易分類」樣本的損失貢獻，使模型更專注於「難分類」的樣本。

#### 訓練引擎

* **評估指標：** 由於類別不平衡，使用 **F1-Score** 作為核心評估指標。
* **學習率調度：** 監控`val_f1_score`，若連續 2 個 epochs 沒有進步，自動降低學習率。
* **提前停止：** 監控`val_f1_score`，若連續 4 個 epochs 沒有進步，自動終止訓練，防止過擬合。
* **最佳模型：** 訓練終止後，腳本會自動載入`val_f1_score`最高的檢查點，並使用測試集進行最終評估。

### 4. 主要設定與超參數 (`src/config.py`)
本專案的核心設定皆集中於`src/config.py`，可依需求自行調整：
* `DEVICE`：自動偵測 cuda 或 cpu。
* `DATA_DIR`：資料集根目錄。
* `IMAGE_SIZE`：影像尺寸（建議 384 以匹配 EffNetV2）。
* `BATCH_SIZE`：批次大小（若 OOM，請調低此值）。
* `EPOCHS`：最大訓練週期。
* `LEARNING_RATE`：初始學習率。
* `EARLY_STOPPING_PATIENCE`：提前停止的耐心值。
* `LORA_RANK`：LoRA 的 Rank。
* `LORA_ALPHA`：LoRA 的 Alpha。
* `FOCAL_ALPHA`, `FOCAL_GAMMA`：Focal Loss 的參數。


### 5. 超參數優化 (`sweep.yaml`)

* 方法：使用貝氏優化`bayes`，根據歷史執行結果推測下一次的最佳參數組合，比隨機搜索 (`random`) 或網格搜索 (`grid`) 更高效。
* 指標：定義優化目標為最大化`val_f1_score`。
* 參數：`LEARNING_RATE`、`LORA_RANK`、`LORA_ALPHA`、`FOCAL_ALPHA`）的最佳組合。