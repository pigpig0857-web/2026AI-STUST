# 把 Label Studio 匯出的 JSON 轉成 YOLO 訓練格式
#
# 兩種讀圖模式：
#   1) 本機模式：LS 就跑在同一台機器上 → 從 LS_MEDIA_DIR 讀（快）
#   2) HTTP 模式：學員從區網下載 JSON 但沒有本機圖檔 → 從 LS_SERVER 用 API token 下載
#
# converter 會先試本機，找不到就 fallback 到 HTTP。

import json
import os
import random
import shutil
from pathlib import Path
from collections import Counter
import urllib.request
import urllib.error

# ====== 學員要改的地方 ======
LS_JSON      = Path(r"C:\Users\TSIC\Downloads\project-1-at-2026-07-08-12-19-054d7225.json")
OUTPUT_DIR   = Path(__file__).parent / "datasets" / "fe"

# 本機模式（老師 / 自己標）：把 LS media 資料夾指過去
LS_MEDIA_DIR = Path(r"C:\Users\TSIC\AppData\Local\label-studio\label-studio\media")

# HTTP 模式（學員從區網 LS 拿資料）：填 server IP + token
# token 拿法：在 LS 網頁 UI 右上頭像 → Account & Settings → Access Token
LS_SERVER    = os.getenv("LS_SERVER", "")           # 例如 "http://192.168.1.102:8081"
LS_TOKEN     = os.getenv("LS_TOKEN",  "")           # 例如 "abc123def456..."

# ====== 拆分比例 ======
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.15
# 剩下 5% 給 test

SEED = 42


def 讀ls_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def 抓所有類別(tasks):
    """掃一遍所有 annotation 取得類別名稱清單（順序穩定）"""
    seen = []
    for t in tasks:
        for a in t.get("annotations", []):
            if a.get("was_cancelled"):
                continue
            for r in a.get("result", []):
                for lbl in r.get("value", {}).get("rectanglelabels", []):
                    if lbl not in seen:
                        seen.append(lbl)
    return seen


def 轉bbox_pct為yolo(x, y, w, h):
    """LS 給的是左上角 x/y + 寬高的百分比 (0-100)
    YOLO 要的是中心 cx/cy + 寬高的比例 (0-1)"""
    cx = (x + w / 2) / 100
    cy = (y + h / 2) / 100
    ww = w / 100
    hh = h / 100
    return cx, cy, ww, hh


def 取得圖片bytes(image_url):
    """
    優先本機讀，找不到就從 LS server 下載
    image_url 例如 '/data/upload/1/xxx.jpg'
    回傳 (bytes, filename) 或 raise
    """
    rel = image_url.split("/data/", 1)[-1]     # upload/1/xxx.jpg
    local = LS_MEDIA_DIR / rel

    # 1) 本機模式
    if local.exists():
        return local.read_bytes(), local.name

    # 2) HTTP fallback
    if not LS_SERVER:
        raise FileNotFoundError(
            f"找不到本機檔案：{local}\n"
            "沒設 LS_SERVER 環境變數，也沒法從 HTTP 下載。\n"
            "設定 LS_SERVER=http://<lS 主機 IP>:8081 和 LS_TOKEN=<你的 token> 再跑一次"
        )

    url = LS_SERVER.rstrip("/") + image_url
    req = urllib.request.Request(url)
    if LS_TOKEN:
        req.add_header("Authorization", f"Token {LS_TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        raise FileNotFoundError(
            f"HTTP {e.code} 從 {url} 下載失敗。"
            + ("Token 可能錯了或過期" if e.code == 401 else "")
        ) from e
    return data, Path(rel).name


def main():
    random.seed(SEED)
    tasks = 讀ls_json(LS_JSON)
    print(f"讀到 {len(tasks)} 個 task")

    classes = 抓所有類別(tasks)
    class_id = {name: i for i, name in enumerate(classes)}
    print(f"類別: {classes}")

    # ====== 準備輸出資料夾 ======
    if OUTPUT_DIR.exists():
        print(f"目標資料夾已存在，刪除後重建：{OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "valid", "test"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # ====== 過濾出有標註的 task ======
    valid_tasks = []
    for t in tasks:
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        # 找 image 欄位（LS 匯出時通常叫 "image"，但你資料可能叫別的）
        data = t.get("data", {})
        image_url = data.get("image") or next(iter(data.values()), None)
        if not image_url:
            continue
        valid_tasks.append((t, anns, image_url))
    print(f"有標註的 task: {len(valid_tasks)}")
    print(f"讀圖模式: {'本機' if LS_MEDIA_DIR.exists() else 'HTTP'} "
          f"({'MEDIA_DIR=' + str(LS_MEDIA_DIR) if LS_MEDIA_DIR.exists() else 'SERVER=' + LS_SERVER})")

    # ====== 隨機拆 train / val / test ======
    random.shuffle(valid_tasks)
    n = len(valid_tasks)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    splits = {
        "train": valid_tasks[:n_train],
        "valid": valid_tasks[n_train:n_train + n_val],
        "test":  valid_tasks[n_train + n_val:],
    }
    for name, items in splits.items():
        print(f"  {name}: {len(items)} 張")

    # ====== 產生檔案 ======
    box_counter = Counter()
    failed = 0
    for split_name, items in splits.items():
        for t, anns, image_url in items:
            # 取圖 bytes（本機或 HTTP）
            try:
                img_bytes, img_name = 取得圖片bytes(image_url)
            except FileNotFoundError as e:
                print(f"  ✗ 跳過 {image_url}: {e}")
                failed += 1
                continue

            dst_img = OUTPUT_DIR / split_name / "images" / img_name
            dst_img.write_bytes(img_bytes)

            # 用 img_name 產出對應 label 檔名
            img_path = Path(img_name)

            # 寫 label
            lines = []
            for a in anns:
                for r in a.get("result", []):
                    if r.get("type") != "rectanglelabels":
                        continue
                    v = r["value"]
                    labels = v.get("rectanglelabels", [])
                    if not labels:
                        continue
                    cls = labels[0]
                    if cls not in class_id:
                        continue
                    cx, cy, ww, hh = 轉bbox_pct為yolo(v["x"], v["y"], v["width"], v["height"])
                    lines.append(f"{class_id[cls]} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
                    box_counter[cls] += 1

            dst_label = OUTPUT_DIR / split_name / "labels" / (img_path.stem + ".txt")
            with open(dst_label, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    # ====== 產生 data.yaml ======
    yaml_content = (
        f"# 由 Label Studio JSON 匯入自動產生\n"
        f"# 產生時間：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"path: {OUTPUT_DIR.resolve().as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\n" + "=" * 50)
    print("完成！")
    print("=" * 50)
    print(f"輸出資料夾: {OUTPUT_DIR}")
    print(f"data.yaml : {yaml_path}")
    print(f"總框數: {sum(box_counter.values())}  ->  {dict(box_counter)}")
    print()
    print("下一步：")
    print(f"1) 改 05_訓練自己的YOLO.py 的 DATA_YAML：")
    print(f"   DATA_YAML = BASE / 'datasets' / 'fe' / 'data.yaml'")
    print(f"2) python 05_訓練自己的YOLO.py")


if __name__ == "__main__":
    main()
