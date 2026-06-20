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
OUT_DIR = ROOT / "report" / "sam_face_instances"
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
    return img.resize((width, int(img.height * width / img.width)), Image.Resampling.LANCZOS)


def prompts(width, height):
    """Tight prompts for visible dolls with a face. Leg-only partial toys are excluded."""
    # name, box normalized, positive points normalized, negative points normalized
    items = [
        ("large_left", (0.10, 0.25, 0.39, 0.62), [(0.23, 0.41), (0.25, 0.52)], [(0.43, 0.44), (0.21, 0.67), (0.09, 0.35)]),
        ("large_center", (0.43, 0.26, 0.71, 0.61), [(0.57, 0.40), (0.56, 0.50)], [(0.40, 0.48), (0.75, 0.50), (0.56, 0.66)]),
        ("front_left", (0.13, 0.47, 0.30, 0.67), [(0.21, 0.55), (0.22, 0.61)], [(0.08, 0.55), (0.34, 0.56), (0.22, 0.70)]),
        ("front_mid_left", (0.31, 0.46, 0.48, 0.68), [(0.39, 0.55), (0.40, 0.61)], [(0.28, 0.55), (0.51, 0.58), (0.40, 0.72)]),
        ("front_mid", (0.48, 0.46, 0.64, 0.68), [(0.56, 0.55), (0.56, 0.61)], [(0.45, 0.56), (0.67, 0.56), (0.56, 0.72)]),
        ("front_mid_right", (0.62, 0.45, 0.78, 0.67), [(0.70, 0.55), (0.70, 0.61)], [(0.59, 0.56), (0.81, 0.56), (0.70, 0.72)]),
        ("front_right", (0.75, 0.47, 0.88, 0.67), [(0.81, 0.56), (0.81, 0.61)], [(0.72, 0.56), (0.91, 0.57), (0.81, 0.72)]),
        ("front_far_right", (0.86, 0.45, 1.00, 0.68), [(0.93, 0.55), (0.93, 0.61)], [(0.83, 0.56), (0.99, 0.46), (0.93, 0.72)]),
        ("small_between_big", (0.31, 0.38, 0.45, 0.56), [(0.38, 0.49), (0.38, 0.54)], [(0.29, 0.49), (0.47, 0.49), (0.38, 0.60)]),
        ("right_cluster_face", (0.75, 0.36, 0.98, 0.58), [(0.89, 0.50), (0.91, 0.55)], [(0.71, 0.47), (0.98, 0.62), (0.86, 0.34)]),
    ]
    scaled = []
    for name, box, pos, neg in items:
        x0, y0, x1, y1 = box
        scaled.append(
            {
                "name": name,
                "box": [int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)],
                "points": [[int(x * width), int(y * height)] for x, y in (pos + neg)],
                "labels": [1] * len(pos) + [0] * len(neg),
            }
        )
    return scaled


def keep_component_near_positive(mask, positives):
    labels, _ = ndi.label(mask)
    keep = np.zeros_like(mask, dtype=bool)
    for x, y in positives:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            lab = labels[y, x]
            if lab:
                keep |= labels == lab
    if keep.any():
        return keep
    ys, xs = np.nonzero(labels)
    if len(xs) == 0:
        return mask
    px, py = positives[0]
    nearest = np.argmin((xs - px) ** 2 + (ys - py) ** 2)
    return labels == labels[ys[nearest], xs[nearest]]


def run_sam(img, prompt_items):
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.eval()
    masks = []
    scores = []

    for item in prompt_items:
        inputs = processor(
            img,
            input_boxes=[[item["box"]]],
            input_points=[[item["points"]]],
            input_labels=[[item["labels"]]],
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=True)
        out_masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        positives = item["points"][: item["labels"].count(1)]
        x0, y0, x1, y1 = item["box"]
        box_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        box_mask = np.zeros(out_masks.shape[-2:], dtype=bool)
        box_mask[y0 : y1 + 1, x0 : x1 + 1] = True

        iou = outputs.iou_scores.cpu()[0, 0]
        candidates = []
        for cand_idx in range(out_masks.shape[1]):
            cand = out_masks[0, cand_idx].numpy().astype(bool)
            cand &= box_mask
            # The assignment is one doll = one visible face. The lower shelf only shows
            # bodies/legs, so it is excluded from this face-instance result.
            cand[int(img.height * 0.635) :, :] = False
            cand = keep_component_near_positive(cand, positives)
            area = int(cand.sum())
            fill = area / box_area
            if area < 600:
                quality = -1
            else:
                # Prefer object-sized masks over tiny mouth/eye fragments.
                quality = float(iou[cand_idx].item()) + min(fill, 0.72) * 1.4
            candidates.append((quality, cand_idx, cand))
        _, best, mask = max(candidates, key=lambda item_: item_[0])
        masks.append(mask)
        scores.append(float(iou[best].item()))
    return masks, scores


def remove_overlaps(masks, scores):
    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
    assigned = np.zeros_like(masks[0], dtype=bool)
    final = [np.zeros_like(masks[0], dtype=bool) for _ in masks]
    for i in order:
        mask = masks[i] & ~assigned
        if mask.sum() < 900:
            continue
        final[i] = mask
        assigned |= mask
    return final


def render_and_cut(img, prompts_, masks, scores):
    arr = np.array(img).astype(np.float32)
    overlay = arr.copy()
    palette = np.array(
        [[235, 86, 92], [63, 145, 220], [55, 175, 120], [244, 178, 52], [156, 102, 226],
         [228, 117, 58], [52, 190, 205], [211, 89, 179], [113, 170, 69], [84, 118, 230]],
        dtype=np.float32,
    )
    rows = []
    crops_dir = OUT_DIR / "face_crops"
    crops_dir.mkdir(exist_ok=True)
    for old in crops_dir.glob("*.png"):
        old.unlink()

    for idx, mask in enumerate(masks, start=1):
        if mask.sum() < 900:
            continue
        color = palette[(idx - 1) % len(palette)]
        overlay[mask] = overlay[mask] * 0.42 + color * 0.58

    result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    thumbs = []
    rgba = np.array(img.convert("RGBA"))

    for idx, mask in enumerate(masks, start=1):
        area = int(mask.sum())
        if area < 900:
            continue
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        color = tuple(int(v) for v in palette[(idx - 1) % len(palette)])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        draw.text((x0 + 4, max(2, y0 - 24)), str(idx), fill=(0, 0, 0), font=FONT_SMALL)

        pad = 10
        x0p, y0p = max(0, x0 - pad), max(0, y0 - pad)
        x1p, y1p = min(img.width - 1, x1 + pad), min(img.height - 1, y1 + pad)
        crop = rgba[y0p : y1p + 1, x0p : x1p + 1].copy()
        alpha = np.zeros(crop.shape[:2], dtype=np.uint8)
        alpha[mask[y0p : y1p + 1, x0p : x1p + 1]] = 255
        crop[:, :, 3] = alpha
        crop_path = crops_dir / f"face_crop_{idx:02d}.png"
        Image.fromarray(crop).save(crop_path)

        thumb = Image.fromarray(crop)
        thumb.thumbnail((175, 170), Image.Resampling.LANCZOS)
        tile = Image.new("RGBA", (200, 220), (255, 255, 255, 255))
        tile.alpha_composite(thumb, ((200 - thumb.width) // 2, 18))
        td = ImageDraw.Draw(tile)
        td.text((8, 185), f"实例 {idx:02d}", fill=(20, 20, 20), font=FONT_SMALL)
        thumbs.append(tile.convert("RGB"))

        rgb = arr[mask].mean(axis=0)
        rows.append(
            {
                "实例编号": idx,
                "名称": prompts_[idx - 1]["name"],
                "SAM置信度": round(scores[idx - 1], 4),
                "面积_像素": area,
                "左上角X": x0,
                "左上角Y": y0,
                "宽度": x1 - x0 + 1,
                "高度": y1 - y0 + 1,
                "平均R": round(float(rgb[0]), 1),
                "平均G": round(float(rgb[1]), 1),
                "平均B": round(float(rgb[2]), 1),
                "裁剪文件": str(crop_path),
            }
        )

    draw.text((18, 18), f"按脸部提示分割实例数：{len(rows)}", fill=(255, 255, 255), font=FONT)
    add_title(result, "SAM 脸部提示实例分割：只保留有脸公仔").save(OUT_DIR / "face_03_instances.jpg", quality=95)

    if thumbs:
        cols = 4
        sheet = Image.new("RGB", (cols * 200, ((len(thumbs) + cols - 1) // cols) * 220 + 56), "white")
        d = ImageDraw.Draw(sheet)
        d.rectangle([0, 0, sheet.width, 56], fill=(24, 38, 60))
        d.text((18, 11), "按脸部提示得到的公仔裁剪图", fill="white", font=FONT)
        for i, tile in enumerate(thumbs):
            sheet.paste(tile, ((i % cols) * 200, 56 + (i // cols) * 220))
        sheet.save(OUT_DIR / "face_04_cutouts.jpg", quality=95)

    df = pd.DataFrame(rows)
    df.to_excel(OUT_DIR / "face_instance_stats.xlsx", index=False)
    return df


def draw_prompts(img, prompt_items):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for idx, item in enumerate(prompt_items, start=1):
        box = item["box"]
        draw.rectangle(box, outline=(0, 220, 255), width=3)
        for point, label in zip(item["points"], item["labels"]):
            color = (255, 55, 55) if label == 1 else (60, 110, 255)
            x, y = point
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color, outline=(255, 255, 255), width=2)
        draw.text((box[0] + 4, max(2, box[1] - 24)), str(idx), fill=(0, 0, 0), font=FONT_SMALL)
    add_title(out, "SAM 精细提示：一张脸/一双眼睛对应一个公仔").save(OUT_DIR / "face_02_prompts.jpg", quality=95)


def main():
    img = resize_to_width(Image.open(SRC_IMAGE).convert("RGB"), 1200)
    prompt_items = prompts(img.width, img.height)
    add_title(img, "原图").save(OUT_DIR / "face_01_original.jpg", quality=95)
    draw_prompts(img, prompt_items)
    masks, scores = run_sam(img, prompt_items)
    masks = remove_overlaps(masks, scores)
    df = render_and_cut(img, prompt_items, masks, scores)
    stats = {
        "method": "sam_face_prompt",
        "model": "facebook/sam-vit-base",
        "prompt_count": len(prompt_items),
        "instance_count": int(len(df)),
        "mean_score": float(df["SAM置信度"].mean()) if len(df) else 0,
        "result_image": str(OUT_DIR / "face_03_instances.jpg"),
        "cutout_contact_sheet": str(OUT_DIR / "face_04_cutouts.jpg"),
        "stats_xlsx": str(OUT_DIR / "face_instance_stats.xlsx"),
    }
    (OUT_DIR / "face_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
