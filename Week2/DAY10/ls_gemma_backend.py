# Label Studio ML Backend：用 gemma4-31b 幫忙預標 bbox
#
# 起服務：
#   set LS_GEMMA_KEY=sk-J19ZkKI3rm8WCj55rZDkUA
#   python ls_gemma_backend.py
# 然後在 LS Settings → Model → Connect Model → http://localhost:9090
#
# 需要：pip install flask requests

import os
import re
import json
import base64
from pathlib import Path

import requests
from flask import Flask, request, jsonify

# ====== 設定 ======
# 執行前先 export：
#   Windows PowerShell:   $env:LS_GEMMA_KEY="sk-xxxxx"
#   Windows cmd:          set LS_GEMMA_KEY=sk-xxxxx
#   Linux/Mac:            export LS_GEMMA_KEY=sk-xxxxx
API_KEY   = os.getenv("LS_GEMMA_KEY")
if not API_KEY:
    raise RuntimeError("請先設定 LS_GEMMA_KEY 環境變數（LiteLLM 的 sk-xxxx）")
GATEWAY   = "https://ai.qianpro.shop/v1"
MODEL     = "gemma4-31b"
# LS 存圖片的本機路徑（我們在同一台機器所以直接讀檔）
LS_MEDIA  = Path(r"C:\Users\TSIC\AppData\Local\label-studio\label-studio\media")

# 你要偵測的類別（要和 LS Labeling Interface 裡的 Label value 完全一致）
CLASSES = ["helmet", "person"]

app = Flask(__name__)


# ====== 呼叫 gemma 拿 bbox ======
def 呼叫gemma(img_b64):
    """把圖丟給 gemma，要求它回 JSON 陣列 [{class, x1,y1,x2,y2}]，座標為 0-100 百分比"""
    prompt = (
        f"你是一個物件偵測助手。仔細看這張圖，找出所有的 {' 和 '.join(CLASSES)}。\n"
        "回傳純 JSON 陣列（不要 markdown code block、不要其他文字）：\n"
        '[{"class":"person","x1":10,"y1":20,"x2":40,"y2":80}, ...]\n'
        "座標範圍 0-100（圖片寬 / 高的百分比）。\n"
        "找不到就回空陣列 [] 。\n"
        f'只能使用這些類別：{CLASSES}'
    )
    r = requests.post(
        f"{GATEWAY}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 1500,
        },
        timeout=90,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]

    # 從回應中挖出 JSON 陣列（gemma 有時候會多加文字）
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        print(f"  gemma 沒吐 JSON：{text[:200]}")
        return []
    try:
        boxes = json.loads(m.group())
        return [b for b in boxes if b.get("class") in CLASSES]
    except json.JSONDecodeError as e:
        print(f"  JSON parse 失敗：{e}\n  原文：{text[:200]}")
        return []


# ====== 把 gemma 的框轉成 LS 需要的 predict 格式 ======
def 轉成LS格式(boxes, img_w, img_h):
    """
    LS 的 rectanglelabels value 用「左上角 x/y 百分比 + 寬高百分比」
    """
    results = []
    for b in boxes:
        x1 = max(0, min(100, float(b["x1"])))
        y1 = max(0, min(100, float(b["y1"])))
        x2 = max(0, min(100, float(b["x2"])))
        y2 = max(0, min(100, float(b["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue
        results.append({
            "from_name": "label",           # 對應 <RectangleLabels name="label" .../>
            "to_name":   "image",           # 對應 <Image name="image" .../>
            "type":      "rectanglelabels",
            "original_width":  img_w,
            "original_height": img_h,
            "image_rotation":  0,
            "value": {
                "x":      x1,
                "y":      y1,
                "width":  x2 - x1,
                "height": y2 - y1,
                "rotation": 0,
                "rectanglelabels": [b["class"]],
            },
        })
    return results


# ====== 讀圖：LS 給的 "/data/upload/1/xxx.jpg" 對到本機路徑 ======
def 讀圖(image_url):
    # 例如 /data/upload/1/002c9ff1-LINE_ALBUM....jpg → upload/1/002c9ff1-....jpg
    rel = image_url.split("/data/", 1)[-1]
    path = LS_MEDIA / rel
    if not path.exists():
        raise FileNotFoundError(f"找不到本機檔案：{path}")
    with open(path, "rb") as f:
        raw = f.read()
    # 拿寬高（用 PIL）
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        w, h = 0, 0
    return base64.b64encode(raw).decode(), w, h


# ====== Label Studio ML Backend 需要的路由 ======
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    tasks = data.get("tasks", [])
    print(f"[predict] 收到 {len(tasks)} 個 task")

    results = []
    for task in tasks:
        try:
            image_url = task["data"]["image"]
            print(f"  → {image_url}")
            img_b64, w, h = 讀圖(image_url)
            boxes = 呼叫gemma(img_b64)
            print(f"    gemma 找到 {len(boxes)} 個框：{[b.get('class') for b in boxes]}")
            ls_result = 轉成LS格式(boxes, w, h)
            results.append({
                "model_version": "gemma4-31b",
                "score": 0.7 if boxes else 0.0,
                "result": ls_result,
            })
        except Exception as e:
            print(f"  預測失敗：{e}")
            results.append({"model_version": "gemma4-31b", "result": [], "score": 0.0})

    return jsonify({"results": results})


@app.route("/health", methods=["GET"])
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL})


@app.route("/setup", methods=["POST"])
def setup():
    # LS 連線時會呼叫這個，回 model_version 就好
    return jsonify({"model_version": "gemma4-31b"})


if __name__ == "__main__":
    print("=" * 50)
    print(f"Gemma ML Backend for Label Studio")
    print(f"Gateway: {GATEWAY}")
    print(f"Model:   {MODEL}")
    print(f"Classes: {CLASSES}")
    print(f"LS media: {LS_MEDIA}")
    print("=" * 50)
    print("Backend URL: http://localhost:9090")
    print("在 LS 裡 Settings → Model → Connect Model 填這個 URL")
    print("=" * 50)
    app.run(host="0.0.0.0", port=9090, threaded=True)
