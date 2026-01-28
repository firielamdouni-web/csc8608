"""
Test rapide pour charger SAM et prédire un masque depuis une bbox.

Usage:
    python TP1/src/quick_test_sam.py --ckpt TP1/models/sam_vit_h_4b8939.pth

Ce script doit être exécuté sur le nœud GPU (conda env avec torch + segment_anything installé).
"""
import argparse
import numpy as np
import cv2
from pathlib import Path
from sam_utils import load_sam_predictor, predict_mask_from_box


def main(ckpt_path: str, model_type: str = "vit_h"):
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        print(f"Checkpoint introuvable: {ckpt}")
        print("Placez le checkpoint dans TP1/models/ ou utilisez download_sam_checkpoint().")
        return

    img_path = next(Path("TP1/data/images").glob("*.jpg"), None)
    if img_path is None:
        print("Aucune image trouvée dans TP1/data/images. Téléchargez d'abord les images.")
        return

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Impossible de lire l'image: {img_path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print("Loading SAM predictor (this may take a while)...")
    predictor = load_sam_predictor(str(ckpt), model_type=model_type)
    print("Predictor loaded.")

    # bbox d'exemple — adaptez si besoin
    H, W = rgb.shape[:2]
    box = np.array([W // 4, H // 4, W * 3 // 4, H * 3 // 4], dtype=np.int32)

    mask, score = predict_mask_from_box(predictor, rgb, box, multimask=True)

    print("img", rgb.shape, "mask", mask.shape, "score", score, "mask_sum", int(mask.sum()))

    # sauvegarder overlay léger
    import os
    from viz_utils import visualize_result

    out_dir = Path("TP1/outputs/overlays")
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"quick_test_{img_path.stem}.png"
    vis = visualize_result(rgb, mask, bbox=box, save_path=str(overlay_path))
    print(f"Overlay saved to {overlay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="TP1/models/sam_vit_h_4b8939.pth")
    parser.add_argument("--model", type=str, default="vit_h")
    args = parser.parse_args()
    main(args.ckpt, args.model)
