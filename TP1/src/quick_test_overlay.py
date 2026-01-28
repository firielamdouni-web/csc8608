"""
Test rapide : exécute SAM sur une image, calcule métriques et génère un overlay.

Usage:
    python TP1/src/quick_test_overlay.py --ckpt TP1/models/sam_vit_h_4b8939.pth

Exécuter sur le nœud GPU avec l'environnement 'deeplearning'.
"""
import argparse
import numpy as np
import cv2
from pathlib import Path
from sam_utils import load_sam_predictor, predict_mask_from_box
from geom_utils import mask_area, mask_bbox, mask_perimeter
from viz_utils import render_overlay


def main(ckpt_path: str, model_type: str = "vit_h"):
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        print(f"Checkpoint introuvable: {ckpt}")
        return

    img_path = next(Path("TP1/data/images").glob("*.jpg"), None)
    if img_path is None:
        print("Aucune image trouvée dans TP1/data/images.")
        return

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    predictor = load_sam_predictor(str(ckpt), model_type=model_type)

    # Exemple de bbox centrée
    H, W = rgb.shape[:2]
    box = np.array([W // 4, H // 4, W * 3 // 4, H * 3 // 4], dtype=np.int32)

    mask, score = predict_mask_from_box(predictor, rgb, box, multimask=True)
    area = mask_area(mask)
    bbox = mask_bbox(mask)
    perim = mask_perimeter(mask)

    overlay = render_overlay(rgb, mask, box, alpha=0.5)
    out_dir = Path("TP1/outputs/overlays")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"overlay_{img_path.stem}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("score", score, "area", area, "bbox", bbox, "perimeter", perim)
    print("saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="TP1/models/sam_vit_h_4b8939.pth")
    parser.add_argument("--model", type=str, default="vit_h")
    args = parser.parse_args()
    main(args.ckpt, args.model)
