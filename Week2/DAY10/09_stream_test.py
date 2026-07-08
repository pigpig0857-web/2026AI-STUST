# CCTV 串流連線測試（純看畫面，不含 YOLO）
#
# 用途：驗證 STREAM_URL 抓對了、cv2.VideoCapture 讀得到
# 通了再用 09_CCTV車流量檢測.py 跑 YOLO 完整版
#
# ⚠ 兩個常見錯誤（oxxostudio 教學版原碼會踩）：
#   1) URL 是 HTML 觀看頁 → cv2 打不開串流。要用 .mpjpeg 直連
#   2) opencv-python-headless 沒 GUI → cv2.imshow 失敗
#      改用 Flask MJPEG 串流到瀏覽器就沒事
#
# ⚠ URL 會過期！台中 CCTV 的 auth token 大概幾小時就換一次
#   過期就重新到觀看頁 F12 抓：
#     https://motoretag.taichung.gov.tw/ATIS_TCC/Device/Showcctv?id=C000002
#     → F12 → Network → 過濾 mpjpeg → 右鍵 Copy URL → 貼下面
#
# 起服務：
#   python 09_stream_test.py
# 瀏覽器：http://localhost:9093/

import time
import threading

import cv2
from flask import Flask, Response, render_template_string

# ====== 改這裡：貼你剛抓的 mpjpeg URL ======
STREAM_URL = "https://tcnvr4.taichung.gov.tw:7001/media/00-0F-7C-13-DA-83.mpjpeg?resolution=240p&auth=cHVibGljdmlld2VyOjYxNmU2NmIxM2RkMjg6YWJiMzZlNzMxNmUzM2M2MjdhMjg5MzY2M2Y4MjhmOGY"


# ====== 抓圖執行緒 ======
最新frame = [None]
執行中 = [True]

def 抓圖迴圈():
    print(f"連線來源：{STREAM_URL[:80]}...")
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print()
        print("×  cv2.VideoCapture 開不了。可能：")
        print("   - URL 是 HTML 頁不是串流本身")
        print("   - Token 過期（重新到觀看頁 F12 抓新的）")
        print("   - 網段不通")
        執行中[0] = False
        return

    print("✓ 串流已開，開始抓 frame")
    frame_count = 0
    while 執行中[0]:
        ret, frame = cap.read()
        if not ret:
            print("frame 讀失敗，重連...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        # 標一下 frame 計數 + 時間，證明真的在跑
        cv2.putText(frame, f"frame {frame_count}  {time.strftime('%H:%M:%S')}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            最新frame[0] = jpg.tobytes()
        frame_count += 1

    cap.release()


threading.Thread(target=抓圖迴圈, daemon=True).start()

# ====== Flask 串流 ======
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Stream Test</title>
<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif;
             display:flex;flex-direction:column;align-items:center;padding:20px}
       img{max-width:100%;border:2px solid #444}</style>
</head><body>
<h1>CCTV 串流測試（無 YOLO）</h1>
<img src="/stream">
</body></html>"""

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
    print("Server: http://localhost:9093/")
    try:
        app.run(host="0.0.0.0", port=9093, threaded=True)
    finally:
        執行中[0] = False
