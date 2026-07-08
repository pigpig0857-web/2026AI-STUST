# 從台中 CCTV 觀看頁自動抓實際 MJPEG 串流 URL
#
# 用法：
#   python get_cctv_url.py C000002
#
# 原理：
#   台中 CCTV 觀看頁 (Showcctv?id=xxx) 底下有一個 <img src="...">
#   那個 img src 就是實際 MJPEG 串流網址（永久 endpoint，不會過期）
#   我們用 urllib GET 那頁的 HTML，用 regex 挖出 img src

import re
import sys
import urllib.request


def 取得CCTV串流URL(device_id: str) -> str:
    viewer_url = f"https://motoretag.taichung.gov.tw/ATIS_TCC/Device/Showcctv?id={device_id}"
    r = urllib.request.urlopen(viewer_url, timeout=15)
    html = r.read().decode("utf-8", errors="replace")

    # 找 <img src="https://tcnvr..."> 或類似的
    m = re.search(r'<img[^>]+src=["\'](https?://tcnvr[^"\']+)["\']', html)
    if not m:
        raise RuntimeError(
            f"CCTV 頁面 {viewer_url} 裡沒找到 stream img。可能：\n"
            "  1) device_id 錯誤（試 C000001、C000003 等）\n"
            "  2) 台中市府網站結構改了"
        )
    return m.group(1)


if __name__ == "__main__":
    device_id = sys.argv[1] if len(sys.argv) > 1 else "C000002"
    url = 取得CCTV串流URL(device_id)
    print(url)
