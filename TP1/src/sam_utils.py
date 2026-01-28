"""
Utilitaires pour charger et utiliser SAM (Segment Anything Model)
"""
import os
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path


def get_device() -> str:
    """Retourne 'cuda' si disponible, sinon 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def load_sam_predictor(checkpoint_path: str, model_type: str = "vit_h") -> SamPredictor:
    """
    Charge SAM et retourne un SamPredictor prêt pour l'inférence.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")
    device = get_device()
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor


@torch.inference_mode()
def predict_mask_from_box(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    multimask: bool = True,
):
    """
    Prévoit un masque à partir d'une bounding box.

    Args:
        predictor: SamPredictor
        image_rgb: (H,W,3) uint8 RGB
        box_xyxy: [x1,y1,x2,y2] en pixels
        multimask: demander plusieurs masques candidates

    Returns:
        mask (H,W) bool, score float
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb doit être (H,W,3)")
    if box_xyxy.shape != (4,):
        raise ValueError("box_xyxy doit être de shape (4,)")

    predictor.set_image(image_rgb)
    box = box_xyxy.astype(np.float32)[None, :]
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=multimask,
    )
    # choisir la meilleure proposition
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(bool)
    score = float(scores[best_idx])
    return mask, score


@torch.inference_mode()
def predict_masks_from_box_and_points(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    point_coords: np.ndarray | None,
    point_labels: np.ndarray | None,
    multimask: bool = True,
):
    """
    Retourne (masks, scores) où :
      - masks : (K, H, W) bool
      - scores : (K,) float
    Supporte une bbox et des points foreground/background facultatifs.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb doit être (H,W,3)")
    if box_xyxy.shape != (4,):
        raise ValueError("box_xyxy doit être de shape (4,)")

    predictor.set_image(image_rgb)
    box = box_xyxy.astype(np.float32)[None, :]
    if point_coords is not None:
        pc = point_coords.astype(np.float32)
        pl = point_labels.astype(np.int64)
    else:
        pc, pl = None, None
    masks, scores, logits = predictor.predict(
        point_coords=pc,
        point_labels=pl,
        box=box,
        multimask_output=multimask,
    )
    return masks.astype(bool), scores.astype(float)


# Backwards-compatible wrappers pour le reste du code
def load_sam_model(model_type="vit_h", checkpoint_path=None, device="cuda"):
    return load_sam_predictor(checkpoint_path, model_type=model_type)


def predict_mask(predictor, image, bbox):
    return predict_mask_from_box(predictor, image, np.array(bbox, dtype=np.int32), multimask=False)


def download_sam_checkpoint(model_type="vit_h", save_dir="./checkpoints"):
    """
    Télécharge le checkpoint SAM si nécessaire
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    checkpoint_names = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }
    checkpoint_path = Path(save_dir) / checkpoint_names[model_type]
    if not checkpoint_path.exists():
        print(f"Téléchargement du checkpoint {model_type}...")
        import urllib.request

        urllib.request.urlretrieve(checkpoint_urls[model_type], checkpoint_path)
        print(f"Checkpoint sauvegardé dans {checkpoint_path}")
    else:
        print(f"Checkpoint déjà présent: {checkpoint_path}")
    return str(checkpoint_path)
