from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from transformers import SamModel, SamProcessor


ROOT = Path(__file__).resolve().parents[1]
SRC_IMAGE = ROOT / "figures" / "source_photo.jpg"
OUT_DIR = ROOT / "report" / "sam_results"
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


def add_title(img, title):
    canvas = Image.new("RGB", (img.width, img.height + 56), "white")
    canvas.paste(img, (0, 56))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, 56], fill=(24, 38, 60))
    draw.text((18, 11), title, fill="white", font=FONT)
    return canvas


def resize_to_width(img, width=1200):
    ratio = width / img.width
    return img.resize((width, int(img.height * ratio)), Image.Resampling.LANCZOS)


def prompt_boxes(width, height):
    """Manual box prompts around the main visible toys."""
    # x0, y0, x1, y1 normalized to the 1200px-wide working image.
    boxes = [
        (0.12, 0.24, 0.36, 0.57),  # large left duck
        (0.43, 0.24, 0.68, 0.56),  # large center duck
        (0.00, 0.34, 0.15, 0.60),  # left edge duck
        (0.75, 0.36, 0.99, 0.60),  # right foreground group
        (0.31, 0.39, 0.45, 0.58),  # mid small
        (0.51, 0.43, 0.62, 0.61),
        (0.62, 0.43, 0.76, 0.62),
        (0.14, 0.50, 0.28, 0.66),
        (0.28, 0.50, 0.40, 0.66),
        (0.43, 0.50, 0.55, 0.66),
        (0.53, 0.50, 0.65, 0.66),
        (0.64, 0.50, 0.76, 0.66),
        (0.75, 0.50, 0.87, 0.66),
        (0.87, 0.50, 0.99, 0.66),
        (0.04, 0.64, 0.17, 0.78),
        (0.20, 0.64, 0.34, 0.78),
        (0.36, 0.64, 0.50, 0.78),
        (0.51, 0.64, 0.65, 0.78),
        (0.66, 0.64, 0.80, 0.78),
        (0.82, 0.64, 0.96, 0.78),
    ]
    return [[int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)] for x0, y0, x1, y1 in boxes]


def draw_boxes(img, boxes):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for idx, box in enumerate(boxes, start=1):
        draw.rectangle(box, outline=(0, 220, 255), width=4)
        draw.text((box[0] + 4, max(2, box[1] - 24)), str(idx), fill=(0, 0, 0), font=FONT_SMALL)
    return add_title(out, "SAM 输入提示：人工框选主要可见玩偶")


def prompt_points(width, height):
    centers = [
        (0.23, 0.40), (0.56, 0.40), (0.05, 0.45), (0.91, 0.52),
        (0.39, 0.52), (0.55, 0.52), (0.69, 0.52),
        (0.21, 0.53), (0.35, 0.53), (0.49, 0.53), (0.60, 0.53),
        (0.72, 0.53), (0.82, 0.53), (0.93, 0.53),
        (0.10, 0.71), (0.27, 0.71), (0.43, 0.71),
        (0.58, 0.71), (0.74, 0.71), (0.90, 0.71),
    ]
    return [[int(x * width), int(y * height)] for x, y in centers]


def keep_component_with_point(mask, point):
    labels, n = ndi.label(mask)
    x, y = point
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return mask
    label = labels[y, x]
    if label == 0:
        # Fall back to the nearest labeled component.
        ys, xs = np.nonzero(labels)
        if len(xs) == 0:
            return mask
        nearest = np.argmin((xs - x) ** 2 + (ys - y) ** 2)
        label = labels[ys[nearest], xs[nearest]]
    return labels == label


def masks_from_sam(img, boxes):
    model_id = "facebook/sam-vit-base"
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id)
    model.eval()

    centers = prompt_points(img.width, img.height)
    point_sets = [[point] for point in centers]
    label_sets = [[1] for _ in centers]
    inputs = processor(img, input_boxes=[boxes], input_points=[point_sets], input_labels=[label_sets], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    scores = outputs.iou_scores.cpu()[0]

    selected = []
    selected_scores = []
    for i in range(masks.shape[0]):
        best = int(torch.argmax(scores[i]).item())
        mask = masks[i, best].numpy().astype(bool)
        box_mask = np.zeros_like(mask, dtype=bool)
        x0, y0, x1, y1 = boxes[i]
        box_mask[y0 : y1 + 1, x0 : x1 + 1] = True
        mask &= box_mask
        mask = keep_component_with_point(mask, centers[i])
        selected.append(mask)
        selected_scores.append(float(scores[i, best].item()))
    return selected, selected_scores


def suppress_overlaps(masks, scores):
    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
    assigned = np.zeros_like(masks[0], dtype=bool)
    final = [np.zeros_like(masks[0], dtype=bool) for _ in masks]
    for idx in order:
        mask = masks[idx] & ~assigned
        if mask.sum() < 450:
            continue
        ys, xs = np.nonzero(mask)
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        aspect = bw / max(1, bh)
        # Drop long shelf strips; plush instances are not extremely horizontal.
        if aspect > 3.2 and bh < 90:
            continue
        final[idx] = mask
        assigned |= mask
    return final


def draw_masks(img, masks, scores):
    arr = np.array(img).astype(np.float32)
    overlay = arr.copy()
    palette = np.array(
        [
            [235, 86, 92], [63, 145, 220], [55, 175, 120], [244, 178, 52],
            [156, 102, 226], [228, 117, 58], [52, 190, 205], [211, 89, 179],
            [113, 170, 69], [84, 118, 230],
        ],
        dtype=np.float32,
    )
    rows = []
    pil = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(pil)

    for idx, mask in enumerate(masks, start=1):
        area = int(mask.sum())
        if area < 450:
            continue
        color = palette[(idx - 1) % len(palette)]
        overlay[mask] = overlay[mask] * 0.48 + color * 0.52
    pil = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)

    for idx, mask in enumerate(masks, start=1):
        area = int(mask.sum())
        if area < 450:
            continue
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        if (x1 - x0 + 1) / max(1, (y1 - y0 + 1)) > 3.2 and (y1 - y0 + 1) < 90:
            continue
        color = tuple(int(v) for v in palette[(idx - 1) % len(palette)])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        draw.text((x0 + 3, max(2, y0 - 24)), str(idx), fill=(0, 0, 0), font=FONT_SMALL)
        rgb = arr[mask].mean(axis=0)
        rows.append(
            {
                "实例编号": idx,
                "SAM置信度": round(scores[idx - 1], 4),
                "面积_像素": area,
                "左上角X": x0,
                "左上角Y": y0,
                "宽度": x1 - x0 + 1,
                "高度": y1 - y0 + 1,
                "平均R": round(float(rgb[0]), 1),
                "平均G": round(float(rgb[1]), 1),
                "平均B": round(float(rgb[2]), 1),
            }
        )

    draw.text((18, 18), f"SAM 框提示实例数：{len(rows)}", fill=(255, 255, 255), font=FONT)
    return add_title(pil, "SAM 框提示实例分割结果"), pd.DataFrame(rows)


def save_cutouts(img, masks, scores):
    crops_dir = OUT_DIR / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for old in crops_dir.glob("sam_crop_*.png"):
        old.unlink()

    rgba = np.array(img.convert("RGBA"))
    rows = []
    thumbs = []
    for idx, mask in enumerate(masks, start=1):
        area = int(mask.sum())
        if area < 450:
            continue
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        if (x1 - x0 + 1) / max(1, (y1 - y0 + 1)) > 3.2 and (y1 - y0 + 1) < 90:
            continue
        pad = 10
        x0p, y0p = max(0, x0 - pad), max(0, y0 - pad)
        x1p, y1p = min(img.width - 1, x1 + pad), min(img.height - 1, y1 + pad)

        crop = rgba[y0p : y1p + 1, x0p : x1p + 1].copy()
        alpha = np.zeros(crop.shape[:2], dtype=np.uint8)
        local_mask = mask[y0p : y1p + 1, x0p : x1p + 1]
        alpha[local_mask] = 255
        crop[:, :, 3] = alpha

        path = crops_dir / f"sam_crop_{idx:02d}.png"
        Image.fromarray(crop).save(path)
        rows.append({"实例编号": idx, "裁剪文件": str(path), "面积_像素": area, "SAM置信度": scores[idx - 1]})

        thumb = Image.fromarray(crop)
        thumb.thumbnail((150, 150), Image.Resampling.LANCZOS)
        tile = Image.new("RGBA", (170, 190), (255, 255, 255, 255))
        tile.alpha_composite(thumb, ((170 - thumb.width) // 2, 18))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 162), f"{idx:02d}", fill=(20, 20, 20), font=FONT_SMALL)
        thumbs.append(tile.convert("RGB"))

    if thumbs:
        cols = 5
        rows_count = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 170, rows_count * 190 + 56), "white")
        draw = ImageDraw.Draw(sheet)
        draw.rectangle([0, 0, sheet.width, 56], fill=(24, 38, 60))
        draw.text((18, 11), "SAM 分割实例透明裁剪总览", fill="white", font=FONT)
        for i, tile in enumerate(thumbs):
            x = (i % cols) * 170
            y = 56 + (i // cols) * 190
            sheet.paste(tile, (x, y))
        sheet.save(OUT_DIR / "sam_04_cutout_contact_sheet.jpg", quality=95)

    pd.DataFrame(rows).to_excel(OUT_DIR / "sam_cutout_files.xlsx", index=False)
    return len(rows), str(crops_dir)


def main():
    img = resize_to_width(Image.open(SRC_IMAGE).convert("RGB"), 1200)
    boxes = prompt_boxes(img.width, img.height)
    add_title(img, "原图：黄色玩偶货架").save(OUT_DIR / "sam_01_original.jpg", quality=95)
    draw_boxes(img, boxes).save(OUT_DIR / "sam_02_prompt_boxes.jpg", quality=95)
    masks, scores = masks_from_sam(img, boxes)
    masks = suppress_overlaps(masks, scores)
    result, df = draw_masks(img, masks, scores)
    result.save(OUT_DIR / "sam_03_instances.jpg", quality=95)
    cutout_count, crops_dir = save_cutouts(img, masks, scores)
    df.to_excel(OUT_DIR / "sam_instance_stats.xlsx", index=False)
    stats = {
        "method": "sam_box_prompt",
        "model": "facebook/sam-vit-base",
        "prompt_count": len(boxes),
        "instance_count": int(len(df)),
        "cutout_count": int(cutout_count),
        "crops_dir": crops_dir,
        "mean_score": float(df["SAM置信度"].mean()) if len(df) else 0,
        "result_image": str(OUT_DIR / "sam_03_instances.jpg"),
        "cutout_contact_sheet": str(OUT_DIR / "sam_04_cutout_contact_sheet.jpg"),
        "stats_xlsx": str(OUT_DIR / "sam_instance_stats.xlsx"),
    }
    (OUT_DIR / "sam_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
