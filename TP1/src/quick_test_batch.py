"""
Process first N images: run SAM bbox->mask, compute metrics, save overlays and print a summary table.

Usage:
    python TP1/src/quick_test_batch.py --ckpt TP1/models/sam_vit_h_4b8939.pth --n 3
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
from sam_utils import load_sam_predictor, predict_mask_from_box
from geom_utils import mask_area, mask_bbox, mask_perimeter
from viz_utils import render_overlay


def main(ckpt_path: str, model_type: str = "vit_h", n: int = 3):
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        print(f"Checkpoint introuvable: {ckpt}")
        return

    imgs = sorted(list(Path("TP1/data/images").glob("*.jpg")))[:n]
    if len(imgs) == 0:
        print("Aucune image trouvée dans TP1/data/images.")
        return

    predictor = load_sam_predictor(str(ckpt), model_type=model_type)

    out_dir = Path("TP1/outputs/overlays")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("image,score,area,perimeter,time_ms,overlay")
    for img_path in imgs:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        box = np.array([W // 4, H // 4, W * 3 // 4, H * 3 // 4], dtype=np.int32)
        import time
        t0 = time.time()
        mask, score = predict_mask_from_box(predictor, rgb, box, multimask=True)
        time_ms = (time.time() - t0) * 1000.0
        area = mask_area(mask)
        perim = mask_perimeter(mask)
        overlay = render_overlay(rgb, mask, box, alpha=0.5)
        out_path = out_dir / f"overlay_{img_path.stem}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"{img_path.name},{score:.6f},{area},{perim:.3f},{time_ms:.1f},{out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="TP1/models/sam_vit_h_4b8939.pth")
    parser.add_argument("--model", type=str, default="vit_h")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()
    main(args.ckpt, args.model, args.n)
