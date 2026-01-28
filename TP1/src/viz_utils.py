"""
Utilitaires pour la visualisation des résultats
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_result(image, mask, bbox=None, save_path=None, alpha=0.5):
    """
    Crée une visualisation avec l'image, le masque en overlay et la bbox
    
    Args:
        image: Image RGB (numpy array)
        mask: Masque binaire
        bbox: Bounding box au format [x_min, y_min, x_max, y_max]
        save_path: Chemin de sauvegarde (optionnel)
        alpha: Transparence du masque (0-1)
    
    Returns:
        numpy array: Image avec overlay
    """
    # TODO: À compléter dans l'exercice 3
    # Créer une copie de l'image
    # result = image.copy()
    
    # Créer un overlay coloré pour le masque
    # overlay = np.zeros_like(image)
    # overlay[mask] = [0, 255, 0]  # Vert
    
    # Fusionner l'image et l'overlay
    # result = cv2.addWeighted(result, 1, overlay, _______, 0)
    
    # Dessiner la bounding box si fournie
    # if bbox is not None:
    #     x_min, y_min, x_max, y_max = bbox
    #     cv2.rectangle(result, (_______, _______), (_______, _______), 
    #                   (255, 0, 0), 2)  # Rouge
    
    # Sauvegarder si demandé
    # if save_path is not None:
    #     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    #     cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    # return result
    pass


def draw_bbox(image, bbox, color=(255, 0, 0), thickness=2, label=None):
    """
    Dessine une bounding box sur une image
    
    Args:
        image: Image (numpy array)
        bbox: [x_min, y_min, x_max, y_max]
        color: Couleur BGR
        thickness: Épaisseur du trait
        label: Label optionnel à afficher
    
    Returns:
        numpy array: Image avec bbox
    """
    result = image.copy()
    x_min, y_min, x_max, y_max = [int(x) for x in bbox]
    
    cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, thickness)
    
    if label is not None:
        cv2.putText(result, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result


def create_comparison_figure(image, mask, bbox, metrics, save_path=None):
    """
    Crée une figure de comparaison avec plusieurs vues
    
    Args:
        image: Image originale
        mask: Masque de segmentation
        bbox: Bounding box
        metrics: Dictionnaire de métriques
        save_path: Chemin de sauvegarde
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image originale avec bbox
    axes[0].imshow(image)
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            fill=False, edgecolor='red', linewidth=2)
        axes[0].add_patch(rect)
    axes[0].set_title("Image originale + BBox")
    axes[0].axis('off')
    
    # Masque seul
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Masque de segmentation")
    axes[1].axis('off')
    
    # Overlay
    overlay = image.copy()
    mask_overlay = np.zeros_like(image)
    mask_overlay[mask] = [0, 255, 0]
    result = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
    axes[2].imshow(result)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    # Ajouter les métriques comme texte
    if metrics:
        metrics_text = f"Aire: {metrics.get('area', 0)} px\n"
        metrics_text += f"Périmètre: {metrics.get('perimeter', 0):.1f}\n"
        bbox_m = metrics.get('bbox', [0, 0, 0, 0])
        metrics_text += f"BBox: {bbox_m[2]}x{bbox_m[3]}"
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def render_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    box_xyxy: np.ndarray,
    alpha: float = 0.5,
):
    """
    Retourne une image RGB uint8 avec :
    - bbox dessinée
    - masque superposé (alpha blending)
    """
    out = image_rgb.copy()
    # 1) Dessiner bbox (en RGB, mais OpenCV dessine en BGR : on dessine sur une version BGR)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = [int(v) for v in box_xyxy.tolist()]
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # 2) Alpha blending du masque (utiliser une couleur fixe, par ex. rouge)
    if mask is not None and mask.any():
        overlay = out.copy().astype(float)
        overlay[mask] = (255, 0, 0)
        out = (alpha * overlay + (1.0 - alpha) * out.astype(float)).astype(np.uint8)
    return out
