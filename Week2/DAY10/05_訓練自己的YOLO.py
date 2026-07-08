# 05 — 訓練你自己的 YOLO11n
#
# 前置：先跑 04 下載 dataset（或手動放到 datasets/<xxx>/data.yaml）
#
# 訓練時間參考（YOLO11n、640x640、20 epochs、剪刀石頭布 1200 張）：
#   - CPU (i5/i7 筆電)          約 30-60 分鐘
#   - GPU (GTX/RTX 消費卡)      約 5-10 分鐘
#   - GPU (RTX 4090 / A100)     1-2 分鐘

from pathlib import Path
import torch
from ultralytics import YOLO

BASE = Path(__file__).parent

# ====== 改這裡：指向你剛下載的 dataset ======
DATA_YAML = BASE / "datasets" / "fe" / "data.yaml"      # 你自己在 LS 標的 Fe 資料
# DATA_YAML = BASE / "datasets" / "rps" / "data.yaml"
# DATA_YAML = BASE / "datasets" / "aquarium" / "data.yaml"

# ====== 訓練參數 ======
EPOCHS      = 20          # 訓練回合。小資料集 20-50，複雜的 100+
IMAGE_SIZE  = 640         # 訓練解析度。越大越準但越慢，通常 320/416/640
BATCH_SIZE  = 8           # GPU 記憶體不夠就調小到 4 或 2
PROJECT_DIR = "runs"      # 訓練結果放這裡
RUN_NAME    = "train"     # 每次會建 runs/train, runs/train2, ...


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"找不到 {DATA_YAML}，請先跑 04_下載範例dataset.py 或手動放資料集"
        )

    has_gpu = torch.cuda.is_available()
    print(f"GPU 可用：{has_gpu}")
    if has_gpu:
        print(f"  device: {torch.cuda.get_device_name(0)}")

    # 從 COCO 預訓練權重開始（transfer learning）
    # 直接寫 'yolo11n.pt' 就會自動下載；若已下載會用本地那份
    model = YOLO("yolo11n.pt")

    results = model.train(
        data       = str(DATA_YAML),
        epochs     = EPOCHS,
        imgsz      = IMAGE_SIZE,
        batch      = BATCH_SIZE,
        device     = 0 if has_gpu else "cpu",
        project    = PROJECT_DIR,
        name       = RUN_NAME,
        exist_ok   = False,     # 每次跑新 folder，不覆蓋
        verbose    = True,
        # 常用可調參數：
        # patience = 10,        # early stop 沒進步就停
        # optimizer = "auto",   # SGD / Adam / AdamW / auto
        # lr0 = 0.01,           # 初始學習率
        # augment = True,       # 用內建資料增強（默認開）
    )

    print()
    print("=" * 60)
    print("訓練完成")
    print("=" * 60)
    print(f"最好的權重：{results.save_dir}/weights/best.pt")
    print(f"最後的權重：{results.save_dir}/weights/last.pt")
    print()
    print("下一步：")
    print("  06_評估訓練結果.py     看 metrics、confusion matrix")
    print("  07_自訓練模型推論.py   把 best.pt 上鏡頭跑跑看")


if __name__ == "__main__":
    main()
