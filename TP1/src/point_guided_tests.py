"""Run automated point-guided experiments on selected images.

Generates overlays for three scenarios per image:
 - bbox only
 - bbox + FG (center)
 - bbox + FG + BG (BG at top-left of bbox)

Saves candidate masks for multimask outputs and writes a CSV summary.
"""
from pathlib import Path
import argparse
import numpy as np
import cv2
import time

from sam_utils import load_sam_predictor, predict_masks_from_box_and_points
from geom_utils import mask_area, mask_bbox, mask_perimeter
from viz_utils import render_overlay


def run_on_image(predictor, img_path: Path, out_dir: Path):
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print('MISSING', img_path)
        return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    box = np.array([W // 4, H // 4, W * 3 // 4, H * 3 // 4], dtype=np.int32)

    scenarios = []

    # Scenario 1: bbox only
    point_coords, point_labels = None, None
    scenarios.append(("bbox_only", point_coords, point_labels))

    # Scenario 2: bbox + FG center
    cx, cy = ( (box[0]+box[2])//2, (box[1]+box[3])//2 )
    point_coords = np.array([(cx, cy)], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int64)
    scenarios.append(("bbox_fg", point_coords, point_labels))

    # Scenario 3: bbox + FG + BG (BG at top-left quarter inside bbox)
    bx1, by1, bx2, by2 = box
    bgx = int(bx1 + (bx2 - bx1) * 0.1)
    bgy = int(by1 + (by2 - by1) * 0.1)
    point_coords = np.array([(cx, cy), (bgx, bgy)], dtype=np.float32)
    point_labels = np.array([1, 0], dtype=np.int64)
    scenarios.append(("bbox_fg_bg", point_coords, point_labels))

    records = []
    for name, pc, pl in scenarios:
        t0 = time.time()
        masks, scores = predict_masks_from_box_and_points(predictor, rgb, box, pc, pl, multimask=True)
        dt = (time.time() - t0) * 1000.0
        # save each candidate mask overlay
        scenario_dir = out_dir / f"{img_path.stem}_{name}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for i in range(masks.shape[0]):
            mask = masks[i].astype(bool)
            overlay = render_overlay(rgb, mask, box, alpha=0.5)
            out_path = scenario_dir / f"candidate_{i}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            area = int(mask_area(mask))
            perim = float(mask_perimeter(mask))
            score = float(scores[i])
            records.append({
                "image": img_path.name,
                "scenario": name,
                "candidate_idx": int(i),
                "score": score,
                "area": area,
                "perimeter": perim,
                "time_ms": float(dt),
                "overlay": str(out_path),
            })

    return records


def main(images, ckpt, out_dir):
    predictor = load_sam_predictor(ckpt)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    for img in images:
        p = Path(img)
        recs = run_on_image(predictor, p, out_dir)
        all_records.extend(recs)

    # write CSV
    import csv
    csv_path = out_dir / "point_guided_results.csv"
    if all_records:
        keys = list(all_records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in all_records:
                writer.writerow(r)
    print("Done. Results saved to", out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs='+', required=True)
    parser.add_argument("--ckpt", type=str, default="TP1/models/sam_vit_h_4b8939.pth")
    parser.add_argument("--out", type=str, default="TP1/outputs/point_guided")
    args = parser.parse_args()
    main(args.images, args.ckpt, args.out)
