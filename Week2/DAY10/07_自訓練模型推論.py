# 07 — 用剛訓練好的 YOLO 模型做 webcam 即時推論
#
# 前置：跑過 05_訓練自己的YOLO.py，會產生 runs/train/weights/best.pt
# 這支腳本自動找最新一個 run 的 best.pt

import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).parent

# ====== 找最新一個 best.pt ======
# Ultralytics 有時把 runs 放在 <DAY10>/runs/，有時放在 project root（跟 python 執行目錄有關）
# 兩個位置都掃，取最新的一份
search_dirs = [BASE / "runs", BASE.parent.parent / "runs"]
weights_candidates = []
for d in search_dirs:
    if d.exists():
        weights_candidates.extend(d.rglob("best.pt"))
weights_candidates = sorted(weights_candidates, key=lambda p: p.stat().st_mtime, reverse=True)
if not weights_candidates:
    raise FileNotFoundError(
        "找不到 best.pt。先跑 05_訓練自己的YOLO.py 完成訓練"
    )
MODEL_PATH = weights_candidates[0]
print(f"載入模型：{MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
print(f"類別名稱：{model.names}")

# ====== 中文字型（自己訓練的類別名如果是中文才用得上）======
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)


def 依類別配色(name):
    h = hash(name) & 0xFFFFFF
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)


def 畫中文標籤(img_bgr, text, xy, size, bg_bgr):
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, size)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    x, y = xy
    bg_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
    draw.rectangle((x, y, x + tw + 6, y + th + 6), fill=bg_rgb)
    draw.text((x + 3, y + 3), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ====== 開 webcam ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("無法開啟 webcam")

print("按 q 離開；按 s 存目前畫面到 snapshot.png")

fps_hist = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    t0 = time.time()

    results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            name  = model.names[cls_id]
            color = 依類別配色(name)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            frame = 畫中文標籤(frame, f"{name} {conf*100:.0f}%",
                              (int(x1), max(0, int(y1) - 30)), 20, color)

    fps_hist.append(1.0 / max(time.time() - t0, 1e-3))
    if len(fps_hist) > 30:
        fps_hist.pop(0)
    fps = sum(fps_hist) / len(fps_hist)
    frame = 畫中文標籤(frame, f"FPS {fps:.1f}  |  {MODEL_PATH.parent.parent.name}",
                     (10, 10), 22, (30, 30, 30))

    cv2.imshow("YOLO 自訓練模型", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        cv2.imwrite("snapshot.png", frame)
        print("已存 snapshot.png")

cap.release()
cv2.destroyAllWindows()
