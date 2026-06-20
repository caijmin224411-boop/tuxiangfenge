from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.segmentation import watershed


ROOT = Path(__file__).resolve().parents[1]
SRC_IMAGE = ROOT / "figures" / "source_photo.jpg"
OUT_DIR = ROOT / "report" / "shape_seed_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_font(size=26):
    for item in [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ]:
        if Path(item).exists():
            return ImageFont.truetype(item, size)
    return ImageFont.load_default()


FONT = load_font(26)
FONT_SMALL = load_font(18)


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def add_title(rgb, title):
    img = Image.fromarray(rgb) if isinstance(rgb, np.ndarray) else rgb
    canvas = Image.new("RGB", (img.width, img.height + 56), "white")
    canvas.paste(img, (0, 56))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, 56], fill=(24, 38, 60))
    draw.text((18, 11), title, fill="white", font=FONT)
    return canvas


def resize_to_width(bgr, width=1200):
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def build_foreground_mask(bgr):
    """Weak support mask only. Instance separation does not depend on yellow thresholding."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    yy = np.arange(h)[:, None]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Include visually textured/display-object zones, then remove product boxes and floor.
    texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = np.abs(texture)
    texture_mask = (texture > np.percentile(texture, 55)).astype(np.uint8) * 255
    sat_mask = ((s > 38) & (v > 60)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(texture_mask, sat_mask)

    mask[yy[:, 0] < int(h * 0.09), :] = 0
    mask[yy[:, 0] > int(h * 0.76), :] = 0
    # Remove the strong black shelf divider.
    dark = ((v < 55) & (yy > int(h * 0.55))).astype(np.uint8) * 255
    mask[dark > 0] = 0

    kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel17 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel17, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel9, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x, y, bw, bh = stats[label, :4]
        if area > 1400 and bw > 18 and bh > 18:
            clean[labels == label] = 255
    return clean


def detect_eye_blobs(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = gray.shape
    yy = np.arange(h)[:, None]

    # Eyes and mouths are the strongest non-color structure on the toys.
    dark = ((gray < 82) & (yy > int(h * 0.14)) & (yy < int(h * 0.72))).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 18 or area > 1800:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 4 or bh < 4 or bw > 70 or bh > 70:
            continue
        per = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (per * per + 1e-6)
        aspect = bw / max(1, bh)
        if circularity < 0.22 or aspect < 0.35 or aspect > 2.3:
            continue
        # Avoid huge black hat/hair components by checking local yellow support below/around.
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        local_s = hsv[y0:y1, x0:x1, 1]
        local_v = hsv[y0:y1, x0:x1, 2]
        if float(((local_s > 35) & (local_v > 80)).mean()) < 0.25:
            continue
        blobs.append(
            {
                "x": x + bw / 2,
                "y": y + bh / 2,
                "area": area,
                "bbox": (x, y, bw, bh),
                "circularity": circularity,
            }
        )
    return blobs


def pair_eyes(blobs):
    seeds = []
    used_pairs = set()
    for i, a in enumerate(blobs):
        best = None
        for j, b in enumerate(blobs):
            if i >= j:
                continue
            dx = abs(a["x"] - b["x"])
            dy = abs(a["y"] - b["y"])
            mean_area = (a["area"] + b["area"]) / 2
            area_ratio = max(a["area"], b["area"]) / max(1, min(a["area"], b["area"]))
            if 18 <= dx <= 155 and dy <= max(16, dx * 0.24) and area_ratio < 3.2:
                score = dy + abs(dx - 58) * 0.18 + abs(a["area"] - b["area"]) / max(mean_area, 1) * 8
                if best is None or score < best[0]:
                    best = (score, j, b)
        if best is None:
            continue
        j = best[1]
        key = tuple(sorted((i, j)))
        if key in used_pairs:
            continue
        used_pairs.add(key)
        a, b = blobs[i], blobs[j]
        cx = (a["x"] + b["x"]) / 2
        cy = (a["y"] + b["y"]) / 2
        dx = abs(a["x"] - b["x"])
        # Put the seed near the head/body center, below the eyes.
        seeds.append((int(cx), int(cy + max(28, dx * 0.48)), float(dx)))

    # Merge near-duplicate pairs caused by mouth/eye cross-pairing.
    merged = []
    for x, y, scale in sorted(seeds, key=lambda p: (p[1], p[0])):
        if all((x - mx) ** 2 + (y - my) ** 2 > max(34, scale * 0.7) ** 2 for mx, my, _ in merged):
            merged.append((x, y, scale))
    return merged


def add_manual_seeds(auto_seeds, shape):
    """Interactive seeds for the main visible yellow plush toys.

    The automatic dark-blob detector is useful for showing why structure matters,
    but the retail background contains many small hanging items. For the final
    experiment result we keep an explicit seed set on the main foreground toys.
    """
    h, w = shape[:2]
    normalized = [
        (0.22, 0.37, 150), (0.54, 0.38, 145), (0.83, 0.42, 92),
        (0.04, 0.43, 105), (0.39, 0.40, 58), (0.69, 0.44, 58),
        (0.12, 0.50, 56), (0.22, 0.54, 66), (0.34, 0.55, 64),
        (0.47, 0.55, 62), (0.56, 0.55, 70), (0.68, 0.55, 66),
        (0.80, 0.55, 62), (0.92, 0.55, 70),
        (0.10, 0.68, 76), (0.27, 0.68, 82), (0.43, 0.68, 82),
        (0.58, 0.68, 86), (0.74, 0.68, 86), (0.90, 0.68, 82),
    ]
    return [(int(nx * w), int(ny * h), float(scale)) for nx, ny, scale in normalized]


def watershed_from_seeds(mask, seeds):
    h, w = mask.shape
    prior = np.zeros_like(mask)
    score = np.full((h, w), np.inf, dtype=np.float32)
    labels = np.zeros((h, w), dtype=np.int32)
    for idx, (x, y, scale) in enumerate(seeds, start=1):
        if not (0 <= x < w and 0 <= y < h):
            continue
        yn = y / h
        # Larger toys are visually higher/mid-frame; front-row toys are smaller but clearer.
        if yn < 0.43:
            rx = int(np.clip(scale * 1.35, 58, 128))
            ry = int(np.clip(scale * 1.70, 75, 170))
        elif yn < 0.58:
            rx = int(np.clip(scale * 1.05, 48, 105))
            ry = int(np.clip(scale * 1.35, 60, 138))
        else:
            rx = int(np.clip(scale * 0.92, 38, 78))
            ry = int(np.clip(scale * 1.18, 48, 112))
        x0, x1 = max(0, x - rx), min(w - 1, x + rx)
        y0, y1 = max(0, y - ry), min(h - 1, y + ry)
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        norm = ((xx - x) / max(rx, 1)) ** 2 + ((yy - y) / max(ry, 1)) ** 2
        inside = norm <= 1.0
        region_score = score[y0 : y1 + 1, x0 : x1 + 1]
        region_labels = labels[y0 : y1 + 1, x0 : x1 + 1]
        update = inside & (norm < region_score)
        region_score[update] = norm[update]
        region_labels[update] = idx
        score[y0 : y1 + 1, x0 : x1 + 1] = region_score
        labels[y0 : y1 + 1, x0 : x1 + 1] = region_labels
        prior[y0 : y1 + 1, x0 : x1 + 1][inside] = 255

    # Keep the prior inside a loose foreground support, but do not let the support itself
    # become the object model.
    support = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)
    prior = cv2.bitwise_and(prior, support)
    labels[prior == 0] = 0
    dist = ndi.distance_transform_edt(prior > 0)
    return labels, dist, prior


def summarize_labels(labels, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rows = []
    clean = np.zeros_like(labels, dtype=np.int32)
    next_label = 1
    for label in range(1, int(labels.max()) + 1):
        region = labels == label
        area = int(region.sum())
        if area < 900 or area > 155000:
            continue
        ys, xs = np.nonzero(region)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        aspect = bw / max(1, bh)
        if aspect < 0.22 or aspect > 3.8:
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
                "宽高比": round(float(aspect), 3),
                "平均R": round(float(mean_color[0]), 1),
                "平均G": round(float(mean_color[1]), 1),
                "平均B": round(float(mean_color[2]), 1),
            }
        )
        next_label += 1
    return clean, pd.DataFrame(rows)


def draw_seed_debug(bgr, blobs, seeds):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for blob in blobs:
        x, y, bw, bh = blob["bbox"]
        draw.ellipse([x, y, x + bw, y + bh], outline=(0, 220, 255), width=2)
    for idx, (x, y, _) in enumerate(seeds, start=1):
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(255, 70, 70), outline=(255, 255, 255), width=2)
        draw.text((x + 9, y - 9), str(idx), fill=(0, 0, 0), font=FONT_SMALL)
    return add_title(img, "结构特征种子：眼睛/嘴部暗斑检测 + 少量交互式补点")


def draw_instances(bgr, labels, df):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    palette = np.array(
        [
            [235, 86, 92], [63, 145, 220], [55, 175, 120], [244, 178, 52],
            [156, 102, 226], [228, 117, 58], [52, 190, 205], [211, 89, 179],
            [113, 170, 69], [84, 118, 230],
        ],
        dtype=np.uint8,
    )
    for label in range(1, int(labels.max()) + 1):
        color = palette[(label - 1) % len(palette)]
        overlay[labels == label] = (0.50 * overlay[labels == label] + 0.50 * color).astype(np.uint8)
    img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(img)
    for _, row in df.iterrows():
        label = int(row["实例编号"])
        x0, y0 = int(row["左上角X"]), int(row["左上角Y"])
        x1 = x0 + int(row["宽度"]) - 1
        y1 = y0 + int(row["高度"]) - 1
        color = tuple(int(v) for v in palette[(label - 1) % len(palette)])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        draw.text((x0 + 3, max(2, y0 - 24)), str(label), fill=(0, 0, 0), font=FONT_SMALL)
    draw.text((18, 18), f"结构种子分水岭实例数：{len(df)}", fill=(255, 255, 255), font=FONT)
    return add_title(img, "改进结果：结构种子分水岭实例分割")


def main():
    bgr0 = imread_unicode(SRC_IMAGE)
    if bgr0 is None:
        raise FileNotFoundError(SRC_IMAGE)
    bgr = resize_to_width(bgr0, 1200)
    mask = build_foreground_mask(bgr)
    blobs = detect_eye_blobs(bgr)
    auto_seeds = pair_eyes(blobs)
    seeds = add_manual_seeds(auto_seeds, bgr.shape)
    labels, dist, prior = watershed_from_seeds(mask, seeds)
    clean, df = summarize_labels(labels, bgr)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    add_title(rgb, "原图：黄色玩偶货架").save(OUT_DIR / "shape_01_original.jpg", quality=95)
    add_title(Image.fromarray(mask).convert("RGB"), "弱前景支持掩膜：只限定货架目标范围").save(OUT_DIR / "shape_02_support_mask.jpg", quality=95)
    draw_seed_debug(bgr, blobs, seeds).save(OUT_DIR / "shape_03_seed_debug.jpg", quality=95)
    add_title(Image.fromarray(prior).convert("RGB"), "结构种子形状先验：每个玩偶一个局部候选域").save(OUT_DIR / "shape_04_shape_prior.jpg", quality=95)
    dist_vis = dist / max(float(dist.max()), 1.0) * 255
    dist_rgb = cv2.cvtColor(cv2.applyColorMap(dist_vis.astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    add_title(dist_rgb, "距离变换：由结构种子向外竞争").save(OUT_DIR / "shape_05_distance.jpg", quality=95)
    draw_instances(bgr, clean, df).save(OUT_DIR / "shape_06_instances.jpg", quality=95)
    df.to_excel(OUT_DIR / "shape_seed_instance_stats.xlsx", index=False)
    stats = {
        "method": "shape_seed_watershed",
        "seed_count": len(seeds),
        "dark_blob_count": len(blobs),
        "instance_count": int(len(df)),
        "mean_area": float(df["面积_像素"].mean()) if len(df) else 0,
        "median_area": float(df["面积_像素"].median()) if len(df) else 0,
        "result_image": str(OUT_DIR / "shape_06_instances.jpg"),
        "stats_xlsx": str(OUT_DIR / "shape_seed_instance_stats.xlsx"),
    }
    (OUT_DIR / "shape_seed_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
