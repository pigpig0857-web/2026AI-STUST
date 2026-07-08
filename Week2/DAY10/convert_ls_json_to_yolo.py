# 把 Label Studio 匯出的 JSON 轉成 YOLO 訓練格式
# 用法：改下面 3 個路徑常數，然後 python convert_ls_json_to_yolo.py

import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict

# ====== 三個要改的路徑 ======
LS_JSON      = Path(r"C:\Users\TSIC\Downloads\project-1-at-2026-07-08-12-19-054d7225.json")
LS_MEDIA_DIR = Path(r"C:\Users\TSIC\AppData\Local\label-studio\label-studio\media")   # LS 圖片存放根目錄
OUTPUT_DIR   = Path(__file__).parent / "datasets" / "fe"                              # 產出到 DAY10/datasets/fe/

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


def 找圖檔實體路徑(image_url):
    """LS 給的 /data/upload/1/xxx.jpg → 真實檔案路徑"""
    rel = image_url.split("/data/", 1)[-1]     # upload/1/xxx.jpg
    return LS_MEDIA_DIR / rel


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

    # ====== 過濾出有標註的 task 且圖檔存在 ======
    valid_tasks = []
    for t in tasks:
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        img_path = 找圖檔實體路徑(t["data"]["image"])
        if not img_path.exists():
            print(f"  跳過（找不到圖檔）：{img_path}")
            continue
        valid_tasks.append((t, anns, img_path))
    print(f"有標註且圖檔存在的 task: {len(valid_tasks)}")

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
    for split_name, items in splits.items():
        for t, anns, img_path in items:
            # 複製圖片
            dst_img = OUTPUT_DIR / split_name / "images" / img_path.name
            shutil.copy2(img_path, dst_img)

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
