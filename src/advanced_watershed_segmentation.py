from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi


ROOT = Path(__file__).resolve().parents[1]
LOCAL_IMAGE = ROOT / "figures" / "source_photo.jpg"
SRC_IMAGE = LOCAL_IMAGE if LOCAL_IMAGE.exists() else Path(
    r"D:\documents\xwechat\xwechat_files\wxid_7v7rttt74gpt11_8f59\temp\RWTemp\2026-06"
    r"\d76229233573c7aad92ba65ef9af4482.jpg"
)
OUT_DIR = ROOT / "segmentation_report_output"
IMG_DIR = OUT_DIR / "figures"
STATS_XLSX = OUT_DIR / "黄色玩偶分水岭实例分割统计.xlsx"
STATS_JSON = OUT_DIR / "advanced_segmentation_stats.json"


def load_font(size=28):
    for item in [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ]:
        if Path(item).exists():
            return ImageFont.truetype(item, size)
    return ImageFont.load_default()


FONT = load_font(24)
FONT_SMALL = load_font(17)


def add_title(img_rgb, title):
    img = Image.fromarray(img_rgb) if isinstance(img_rgb, np.ndarray) else img_rgb
    canvas = Image.new("RGB", (img.width, img.height + 54), "white")
    canvas.paste(img, (0, 54))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, 54], fill=(30, 47, 75))
    draw.text((18, 10), title, fill="white", font=FONT)
    return canvas


def resize_image(bgr, width=1200):
    h, w = bgr.shape[:2]
    ratio = width / w
    return cv2.resize(bgr, (width, int(h * ratio)), interpolation=cv2.INTER_AREA)


def build_refined_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    height = bgr.shape[0]
    roi = np.zeros((height, bgr.shape[1]), dtype=np.uint8)
    roi[int(height * 0.16) :, :] = 255

    yellow = cv2.inRange(hsv, np.array([13, 55, 95]), np.array([39, 255, 255]))
    orange = cv2.inRange(hsv, np.array([4, 70, 80]), np.array([24, 255, 255]))
    warm_lab = cv2.inRange(lab, np.array([120, 124, 138]), np.array([255, 165, 220]))

    white_box = ((s < 48) & (v > 145) & (np.indices(mask_shape := h.shape)[0] > int(height * 0.72))).astype(np.uint8) * 255
    dark_floor = ((v < 55) & (np.indices(mask_shape)[0] > int(height * 0.63))).astype(np.uint8) * 255
    dark_hat = ((v < 75) & (np.indices(mask_shape)[0] < int(height * 0.42))).astype(np.uint8) * 255

    mask = cv2.bitwise_or(yellow, orange)
    mask = cv2.bitwise_or(mask, warm_lab)
    mask = cv2.bitwise_and(mask, roi)
    mask[white_box > 0] = 0
    mask[dark_floor > 0] = 0
    mask[dark_hat > 0] = 0

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)
    return mask


def watershed_instances(mask):
    distance = ndi.distance_transform_edt(mask > 0)
    coords = peak_local_max(
        distance,
        min_distance=28,
        threshold_abs=12,
        labels=mask > 0,
        exclude_border=False,
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    for idx, (y, x) in enumerate(coords, start=1):
        markers[y, x] = idx
    markers = ndi.label(markers > 0)[0]
    labels = watershed(-distance, markers, mask=mask > 0)
    return labels, distance


def filter_instances(labels, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rows = []
    clean = np.zeros_like(labels, dtype=np.int32)
    next_label = 1
    for label in range(1, int(labels.max()) + 1):
        region = labels == label
        area = int(region.sum())
        if area < 260 or area > 25000:
            continue
        ys, xs = np.nonzero(region)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        aspect = bw / max(1, bh)
        if aspect < 0.15 or aspect > 4.6:
            continue
        mean_color = rgb[region].mean(axis=0)
        clean[region] = next_label
        rows.append(
            {
                "实例编号": next_label,
                "面积_像素": area,
                "左上角X": x0,
                "左上角Y": y0,
                "宽度": bw,
                "高度": bh,
                "宽高比": round(aspect, 3),
                "平均R": round(float(mean_color[0]), 1),
                "平均G": round(float(mean_color[1]), 1),
                "平均B": round(float(mean_color[2]), 1),
            }
        )
        next_label += 1
    return clean, pd.DataFrame(rows)


def draw_instances(bgr, labels, df):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    palette = np.array(
        [
            [236, 88, 128],
            [59, 151, 225],
            [244, 183, 64],
            [75, 185, 133],
            [157, 108, 230],
            [233, 112, 71],
            [52, 190, 205],
            [215, 84, 198],
            [100, 180, 80],
        ],
        dtype=np.uint8,
    )
    for label in range(1, int(labels.max()) + 1):
        color = palette[(label - 1) % len(palette)]
        overlay[labels == label] = (0.58 * overlay[labels == label] + 0.42 * color).astype(np.uint8)
    pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)
    for _, row in df.iterrows():
        x0, y0 = int(row["左上角X"]), int(row["左上角Y"])
        x1 = x0 + int(row["宽度"]) - 1
        y1 = y0 + int(row["高度"]) - 1
        color = tuple(int(v) for v in palette[(int(row["实例编号"]) - 1) % len(palette)])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        if int(row["面积_像素"]) > 450:
            draw.text((x0 + 2, max(2, y0 - 22)), str(int(row["实例编号"])), fill=(0, 0, 0), font=FONT_SMALL)
    draw.text((18, 18), f"OpenCV + skimage 分水岭实例数：{len(df)}", fill=(255, 255, 255), font=FONT)
    return add_title(pil, "图10 OpenCV+skimage 黄色玩偶分水岭进一步分割")


def save_distance(distance, path):
    dist = distance.copy()
    if dist.max() > 0:
        dist = dist / dist.max() * 255
    colored = cv2.applyColorMap(dist.astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    add_title(Image.fromarray(rgb), "图9 距离变换热力图：亮处更接近目标中心").save(path, quality=95)


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    bgr0 = cv2.imdecode(np.fromfile(str(SRC_IMAGE), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr0 is None:
        raise FileNotFoundError(SRC_IMAGE)
    bgr = resize_image(bgr0)
    mask = build_refined_mask(bgr)
    labels, distance = watershed_instances(mask)
    clean, df = filter_instances(labels, bgr)

    mask_path = IMG_DIR / "fig08_refined_cv_mask.jpg"
    dist_path = IMG_DIR / "fig09_distance_transform.jpg"
    ws_path = IMG_DIR / "fig10_watershed_instances.jpg"
    add_title(Image.fromarray(mask).convert("RGB"), "图8 OpenCV 黄色/橙色精细掩膜").save(mask_path, quality=95)
    save_distance(distance, dist_path)
    draw_instances(bgr, clean, df).save(ws_path, quality=95)

    df.to_excel(STATS_XLSX, index=False)
    stats = {
        "mask_path": str(mask_path),
        "distance_path": str(dist_path),
        "watershed_path": str(ws_path),
        "stats_xlsx": str(STATS_XLSX),
        "instance_count": int(len(df)),
        "mean_area": float(df["面积_像素"].mean()) if len(df) else 0,
        "median_area": float(df["面积_像素"].median()) if len(df) else 0,
        "total_area": int(df["面积_像素"].sum()) if len(df) else 0,
    }
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
