# 09 — CCTV 車流量檢測
#
# 讀取台中市政府公開 CCTV 串流（或任何 MJPEG / RTSP / HTTP 影片）
# 用 YOLO11n（COCO 預訓練）偵測 車 / 卡車 / 機車 / 公車
# ByteTrack 幫每台車一個 ID，過中線就 +1，區分方向（上行/下行）
# 畫面串流到瀏覽器，同 WiFi 都能看
#
# 起服務：
#   python 09_CCTV車流量檢測.py
# 瀏覽器：
#   http://localhost:9092/  或  http://192.168.1.102:9092/

import os
import time
import threading
from collections import defaultdict

import cv2
import numpy as np
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ====== 串流來源 ======
# 用 get_cctv_url.py 自動從觀看頁抓，永久有效
from get_cctv_url import 取得CCTV串流URL
DEVICE_ID  = os.getenv("DEVICE_ID", "C000002")
STREAM_URL = 取得CCTV串流URL(DEVICE_ID)
print(f"CCTV device {DEVICE_ID} → {STREAM_URL}")

# ====== YOLO 設定 ======
MODEL_PATH   = "yolo11n.pt"       # 沒有會自動下載 5.5MB
CONF_THRESH  = 0.35

# COCO 的車輛類別 ID
VEHICLE_IDS = {
    2: ("car",        "汽車", (100, 200, 255)),
    3: ("motorcycle", "機車", (100, 255, 100)),
    5: ("bus",        "公車", (255, 200, 100)),
    7: ("truck",      "卡車", (255, 100, 200)),
}

# ====== 中文字型 ======
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)

def 畫中文(img_bgr, text, xy, size, color_bgr, bg=True):
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, size)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    x, y = xy
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    if bg:
        draw.rectangle((x, y, x + tw + 6, y + th + 6), fill=color_rgb)
        draw.text((x + 3, y + 3), text, font=font, fill=(255, 255, 255))
    else:
        draw.text((x, y), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ====== 模型 ======
print(f"載入 {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# ====== 執行緒共享狀態 ======
最新frame = [None]
執行中 = [True]
fps_hist = []
上行總數 = [0]
下行總數 = [0]
per_class_count = defaultdict(int)   # {class_id: 該類累計數}
已計數ID = set()
track_歷史 = {}                     # {tid: (prev_x, prev_y)}

def 開啟來源():
    """優先用 STREAM_URL，失敗就 fallback 到 webcam"""
    src = STREAM_URL
    # 允許 STREAM_URL=0/1 這種數字表示 webcam
    if src.isdigit():
        src = int(src)
        print(f"用 webcam {src}")
    else:
        print(f"嘗試連線串流：{src[:80]}...")

    cap = cv2.VideoCapture(src)
    if cap.isOpened():
        return cap, src

    print()
    print("=" * 60)
    print("串流連不上，可能原因：")
    print("  1) auth token 過期 — 台中 CCTV 每個 session 換 token")
    print("  2) 網段不通、DNS 錯誤、被防火牆擋")
    print()
    print("怎麼拿新的 URL：")
    print("  1) 瀏覽器打開 https://motoretag.taichung.gov.tw/ATIS_TCC/Device/Showcctv?id=C000002")
    print("  2) F12 → Network tab → 重整頁面")
    print("  3) 過濾 mpjpeg → 找到 request 右鍵 Copy URL")
    print("  4) 重跑：$env:STREAM_URL='貼新URL'; python 09_CCTV車流量檢測.py")
    print()
    print("Fallback: 改用 webcam...")
    print("=" * 60)
    cap = cv2.VideoCapture(0)
    return cap, 0


def 抓圖與推論():
    cap, src = 開啟來源()
    if not cap.isOpened():
        print("Webcam 也開不了，放棄。")
        執行中[0] = False
        return

    line_y = None   # 過線 y 座標，第一 frame 後才算

    while 執行中[0]:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("串流中斷，重連...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(src)
            continue

        H, W = frame.shape[:2]
        if line_y is None:
            line_y = int(H * 0.55)   # 中線放在畫面 55% 高的位置

        # YOLO 追蹤（persist=True 讓 track ID 跨 frame 維持）
        results = model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_IDS.keys()),
            conf=CONF_THRESH,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        # 畫每輛車
        現有車數 = defaultdict(int)
        for r in results:
            boxes = r.boxes
            if boxes is None or boxes.id is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cls_id = int(boxes.cls[i])
                conf   = float(boxes.conf[i])
                tid    = int(boxes.id[i])

                if cls_id not in VEHICLE_IDS:
                    continue
                _, name_zh, color = VEHICLE_IDS[cls_id]
                現有車數[cls_id] += 1

                # 畫框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                frame = 畫中文(frame, f"#{tid} {name_zh} {conf*100:.0f}%",
                              (int(x1), max(0, int(y1) - 26)), 16, color)

                # 中心點
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 3, color, -1)

                # 過線判定
                if tid in track_歷史:
                    prev_y = track_歷史[tid][1]
                    if tid not in 已計數ID:
                        # 由上往下穿過中線 → 下行
                        if prev_y < line_y <= cy:
                            下行總數[0] += 1
                            per_class_count[cls_id] += 1
                            已計數ID.add(tid)
                        # 由下往上穿過中線 → 上行
                        elif prev_y > line_y >= cy:
                            上行總數[0] += 1
                            per_class_count[cls_id] += 1
                            已計數ID.add(tid)
                track_歷史[tid] = (cx, cy)

        # 清舊 track（避免記憶體長期漲）
        if len(track_歷史) > 500:
            keep_ids = set(list(track_歷史.keys())[-300:])
            for k in list(track_歷史.keys()):
                if k not in keep_ids:
                    track_歷史.pop(k, None)
                    已計數ID.discard(k)

        # 畫計數線
        cv2.line(frame, (0, line_y), (W, line_y), (0, 255, 255), 2)
        frame = 畫中文(frame, "計數線", (10, line_y + 4), 16, (0, 255, 255), bg=False)

        # 上方 HUD
        fps_hist.append(1.0 / max(time.time() - t0, 1e-3))
        if len(fps_hist) > 30:
            fps_hist.pop(0)
        fps = sum(fps_hist) / len(fps_hist)

        hud = f"FPS {fps:.1f}  |  ↓下行 {下行總數[0]}  |  ↑上行 {上行總數[0]}"
        frame = 畫中文(frame, hud, (10, 10), 22, (30, 30, 30))

        # 右上：現有車輛數 by 類別
        y = 40
        for cid, count in 現有車數.items():
            _, name_zh, color = VEHICLE_IDS[cid]
            frame = 畫中文(frame, f"{name_zh}: 現有 {count}  累計 {per_class_count[cid]}",
                          (W - 260, y), 16, color)
            y += 22

        # 編成 JPEG 給 Flask
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            最新frame[0] = jpg.tobytes()

    cap.release()


threading.Thread(target=抓圖與推論, daemon=True).start()

# ====== Flask 串流 ======
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>CCTV 車流量檢測</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:'Microsoft JhengHei',sans-serif;
         display:flex; flex-direction:column; align-items:center; padding:20px; }
  h1 { margin:8px 0; }
  .info { color:#aaa; font-size:14px; margin-bottom:12px; text-align:center; }
  img { max-width:100%; border:2px solid #444; border-radius:8px; }
  code { background:#333; padding:2px 6px; border-radius:3px; color:#7cf; }
</style>
</head>
<body>
  <h1>CCTV 車流量即時檢測</h1>
  <div class="info">
    <div>YOLO11n · ByteTrack · 過中線計數</div>
    <div>類別：汽車 / 機車 / 公車 / 卡車</div>
  </div>
  <img src="/stream">
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

def mjpeg():
    while True:
        f = 最新frame[0]
        if f is None:
            time.sleep(0.1)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(f)).encode() + b"\r\n\r\n" + f + b"\r\n")

@app.route("/stream")
def stream():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("=" * 60)
    print("CCTV 車流量檢測 - 瀏覽器串流版")
    print("=" * 60)
    print("本機打:      http://localhost:9092/")
    print("學員同 WiFi: http://192.168.1.102:9092/")
    print("Ctrl+C 停止")
    print("=" * 60)
    try:
        app.run(host="0.0.0.0", port=9092, threaded=True)
    finally:
        執行中[0] = False
