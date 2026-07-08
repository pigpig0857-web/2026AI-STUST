# YOLO 訓練指南（不同環境 / 硬體 / 路徑都能跑）

DAY10 的訓練腳本 (`05_訓練自己的YOLO.py`) 已經寫成「自動偵測」版本，但每個學員的環境還是不太一樣。這份文件收錄各種情境的注意事項與踩雷經驗。

---

## 1. 環境檢查清單（每個學員都要做）

跑訓練前，PowerShell 執行：

```powershell
conda activate STUSTPython
cd C:\Users\<你的名字>\Documents\project    # 路徑改成你自己的
python -c "import torch, ultralytics; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__)"
```

輸出應該長這樣：

```
torch 2.12.1
cuda True   ← 或 False
ultralytics 8.4.90
```

如果任何一項 `ImportError` → 沒裝：

```powershell
pip install -r Week2\DAY10\requirements.txt
```

---

## 2. 依你的硬體對應設定

### 情境 A：完全沒 GPU（Mac、筆電）

**能不能訓練**：能。只是慢。

**參數建議**（改 `05_訓練自己的YOLO.py`）：

```python
EPOCHS       = 20      # 別加太多，跑不完
IMAGE_SIZE   = 320     # 從 640 降到 320，速度快 4 倍，準度掉一點
# batch 由 script 自動決定，CPU 會用 8
```

**訓練時間**：
- 140 張圖 20 epoch：**約 15-30 分鐘**
- 1000 張圖 20 epoch：**約 1-2 小時**
- 5000 張圖 20 epoch：**放棄，改上 Colab**

**踩雷提示**：
- 訓練期間 CPU 100%，電腦會**熱、慢、風扇狂轉** — 別同時開 Chrome 100 分頁
- 筆電**接電源**，別靠電池訓練（降頻)

### 情境 B：消費級 GPU（GTX 1660 / RTX 3060 / 4060 / 4070）

**能不能訓練**：能。速度好。

**參數建議**：
```python
EPOCHS       = 50      # 可以放心加，還是 30 分內
IMAGE_SIZE   = 640     # 用預設
# batch 由 script 依顯存自動：6GB→8、12GB→16
```

**訓練時間**：
- 140 張 50 epoch：**3-6 分鐘**
- 1000 張 50 epoch：**15-30 分鐘**

**踩雷提示**：
- 用完關掉遊戲、Chrome、任何 GPU 加速的東西
- OOM (Out Of Memory) 時：script 建議的 batch 太大，手動改小一階：8 → 4 → 2

### 情境 C：工作站級 GPU（RTX 3090 / 4090 / A6000）

**能不能訓練**：能，用手撕開，快到不夠嗨。

**參數建議**：
```python
EPOCHS       = 100
IMAGE_SIZE   = 640
# batch 由 script 用 32
```

**訓練時間**：140 張 100 epoch **90 秒內**。

**建議加碼**：
- 換更大模型：`YOLO("yolo11s.pt")`（21MB）甚至 `yolo11m.pt`（40MB）
- 開 augmentation 更豐富：`model.train(..., mosaic=1.0, mixup=0.5, ...)`

### 情境 D：無 GPU 但想快 → 用 Google Colab（免費）

**適用**：完全沒 GPU 或想跑大資料集。

**步驟**：
1. Colab 打開新 notebook：<https://colab.research.google.com/>
2. 上方選單 → `Runtime` → `Change runtime type` → `T4 GPU`
3. 執行：

```python
!pip install ultralytics
from google.colab import files
uploaded = files.upload()   # 上傳你的 data.yaml + images/ + labels/（壓縮成 zip 上傳最快）

!unzip yourdataset.zip -d /content/data
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="/content/data/data.yaml", epochs=50, imgsz=640, batch=16)
```

4. 訓練完載回權重：
```python
files.download("/content/runs/detect/train/weights/best.pt")
```
把 `best.pt` 放到本機 `Week2/DAY10/runs/detect/train/weights/best.pt` 就能給 07 / 08 用。

---

## 3. 路徑差異（每個學員的 project 位置不一樣）

### 學員甲：`C:\Users\Alice\Documents\project\`
### 學員乙：`D:\ai-course\2026-project\`
### 學員丙：Mac `/Users/bob/Projects/stust-ai/`

腳本已經處理好，因為所有路徑用：

```python
BASE = Path(__file__).parent      # 05_訓練自己的YOLO.py 所在資料夾
DATASETS = BASE / "datasets"       # 相對於腳本自己
PROJECT_DIR = BASE / "runs"        # 相對於腳本自己（絕對路徑）
```

**不管你在哪 `cd`，也不管 project 放在哪**，訓練結果一定在 `DAY10/runs/`。

### 踩雷：不要在 project root 亂 `cd`
以前 05 用相對路徑 `PROJECT_DIR = "runs"`，導致：
- 你在 project root 執行 → 存到 `project/runs/`
- 你在 DAY10 執行 → 存到 `DAY10/runs/`

新版已修正，不會再有這問題。

---

## 4. 資料集情境

### 情境 A：跑範例（Roboflow 下載的）
```powershell
python Week2\DAY10\04_下載範例dataset.py
# 修改 use_this = "rps"（或 hardhat、person...）
```
下載完直接跑 05，會被 script 自動偵測到。

### 情境 B：自己在 Label Studio 標

**兩種子情境**：

#### B-1：老師 or 自己標，LS 跟訓練在同一台機器
```powershell
# LS 匯出 JSON → 改 convert 腳本三個路徑
#   LS_JSON      = 你下載的 .json 檔案
#   LS_MEDIA_DIR = LS 存圖的資料夾（Windows 通常在 %LOCALAPPDATA%\label-studio\label-studio\media）
#   OUTPUT_DIR   = datasets/<你想要的名字>
python Week2\DAY10\convert_ls_json_to_yolo.py
```

#### B-2：學員從**區網**下載 LS 匯出的 JSON（老師的 LS 在另一台）
LS 的 YOLO 匯出**不含原始圖檔**（只給標籤 + URL），學員自己那台沒有圖。
**Converter 已經支援 HTTP fallback**：找不到本機檔就從老師的 LS server 下載。

**Step 1**：學員登入 LS UI → 右上頭像 → `Account & Settings` → 找 `Access Token` 複製

**Step 2**：學員在自己電腦 PowerShell：
```powershell
$env:LS_SERVER="http://192.168.1.102:8081"   # 老師的 LS server (跟老師確認 IP)
$env:LS_TOKEN="複製那串很長的 token"
python Week2\DAY10\convert_ls_json_to_yolo.py
```

Converter 會逐張透過 HTTP 從老師的 LS 下載圖 + 存到 `datasets/xxx/`。
- 140 張圖大約 30-60 秒（同 WiFi）
- 學員不用管圖檔在哪，一氣呵成

**Step 3**：`data.yaml`、`train/val/test` 拆分 converter 都幫你做好，直接開跑 `05`。

### 情境 C：資料集在雲端（NAS / Google Drive）
把 `data.yaml` 內的 `path:` 改成絕對路徑，例如：
```yaml
path: D:/shared/datasets/factory-defect
train: train/images
val: valid/images
```
腳本會直接用。

### 情境 D：沒圖不會標
最簡單：跑 04 下載 **Rock-Paper-Scissors** 或 **Hard Hat Workers** 練手。這兩個是 Roboflow 上最常用的教學資料集，5 分鐘下完。

---

## 5. 常見錯誤（依訊息查）

### `CUDA out of memory`
GPU 顯存爆了。
- 手動改 05 的 batch size：8 → 4 → 2
- 或降低 IMAGE_SIZE：640 → 480 → 320
- 關掉背景吃 GPU 的程式（遊戲、瀏覽器硬體加速）

### `No module named 'ultralytics'`
沒裝或裝到錯的 env。
```powershell
conda activate STUSTPython
pip install ultralytics
```

### `[WinError 32] The process cannot access the file... yolo11n.pt`
之前訓練沒關乾淨還鎖在 memory。
```powershell
taskkill /F /IM python.exe
# 重來
```

### 訓練跑到一半停住不動
- CPU：正常，只是慢
- GPU：可能被別的程式搶走。看工作管理員 → 效能 → GPU 使用率
- 或是 dataloader 卡住：把 05 加 `workers=0`（單線程 loader，慢但穩）

### `cv2.error: The function is not implemented`（`imshow` 相關）
opencv-python-headless 被裝了。
- 治本：`pip uninstall opencv-python-headless -y; pip install opencv-python`
- 治標：**用 08 串流版**，不需要 GUI opencv

### `FileNotFoundError: 找不到 best.pt`
訓練還沒完成，或存到怪地方。
- 檢查 `runs/train/weights/` 有沒有檔案
- 07/08 都自動掃 `DAY10/runs/` 跟 `project/runs/` 兩個位置，訓練完會找到

### `mAP@0.5 一直 0.0` 或 訓練都沒進步
- 檢查 `data.yaml` 的 `names:` 跟你標的類別對不對
- 檢查 `labels/*.txt` 檔案有沒有東西（不是空的）
- 用 `06_評估訓練結果.py` 看 loss 有沒有降；沒降代表資料出錯

### 訓練完 `07` 打不開 webcam
- 別人的視訊軟體佔用（Zoom、Teams、Chrome LS 頁面）
- 檢查 07 開頭 `cv2.VideoCapture(0)`，0 是第一支鏡頭；有多支的話試 `1`, `2`
- 換用 `08` 串流版，`http://localhost:9091/` 開瀏覽器看

---

## 6. 訓練參數調整參考

改 `05_訓練自己的YOLO.py` 上方的常數：

| 參數 | 預設 | 什麼時候要改 |
|---|---|---|
| `EPOCHS` | 20 | 資料量 > 2000 張 → 加到 50-100 |
| `IMAGE_SIZE` | 640 | CPU 訓練 → 降到 320；有 GPU 且想極致準 → 加到 1280 |
| `PATIENCE` | 10 | 資料量小時可以降到 5，早停避免過擬合 |

model.train 內還有很多可調，去 <https://docs.ultralytics.com/modes/train/#train-settings> 查。

---

## 7. 訓練後怎麼判斷有沒有訓好

跑 `06_評估訓練結果.py` 看 4 張圖：

| 指標 | 好的樣子 | 壞的樣子 |
|---|---|---|
| **Loss** | 前 5 epoch 快速下降，之後平緩 | 一直維持高 → 資料錯 or 學不到 |
| **mAP@0.5** | 訓練 & 驗證都上升，最後 > 0.7 | 訓練上升驗證不動 → 過擬合 |
| **Precision** | > 0.7 | < 0.5 → 太多假警報 |
| **Recall** | > 0.7 | < 0.5 → 太多漏抓 |

如果不理想，回頭做這幾件事：
1. 資料量太少 → 標更多，尤其少的類別
2. 資料太乾淨 → augmentation 加大（`mosaic`, `mixup`, `hsv_h`, `degrees` 都可以調）
3. 資料標錯 → 用 06 產出的 `val_batch0_pred.jpg` 對照，看哪些框標歪了
4. 類別不平衡 → 補資料，或用 `class_weights`
5. Backbone 太小 → 換 `yolo11s.pt`（21MB）試試

---

## 8. 一個學員的完整流程 example

**小明**：Windows 桌機、RTX 3060 12GB、有 200 張自拍照片要辨識自己臉。

```powershell
# 1) 環境
conda activate STUSTPython
cd C:\Users\小明\Documents\project

# 2) 資料 — 用 Label Studio 標完 200 張，匯出 JSON
#    改 convert_ls_json_to_yolo.py 上方三個路徑
python Week2\DAY10\convert_ls_json_to_yolo.py
# → 產生 Week2\DAY10\datasets\my_face\

# 3) 訓練
python Week2\DAY10\05_訓練自己的YOLO.py
# script 自動偵測到 datasets/my_face，選 1 進入
# 提示：偵測到 GPU (RTX 3060 12.0 GB)、建議 batch=16
# 按 Enter 開始
# 3060 上訓練 50 epoch 約 15 分鐘

# 4) 看結果
python Week2\DAY10\06_評估訓練結果.py

# 5) 上鏡頭驗證
python Week2\DAY10\08_webcam串流推論.py
# 瀏覽器打 http://localhost:9091/
```

**結束。** 全程沒改路徑、不寫死名字，換到任何一台電腦流程一樣。
