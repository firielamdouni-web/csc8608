"""
Utilitaires pour les calculs géométriques sur les masques
"""
import numpy as np
import cv2


def compute_mask_metrics(mask):
    """
    Calcule les métriques géométriques d'un masque de segmentation
    
    Args:
        mask: Masque binaire (numpy array booléen ou uint8)
    
    Returns:
        dict: Dictionnaire contenant les métriques
            - area: Aire en pixels
            - perimeter: Périmètre approximatif
            - bbox: Bounding box du masque [x, y, w, h]
            - centroid: Centre de masse [cx, cy]
    """
    # TODO: À compléter dans l'exercice 3
    # Convertir en uint8 si nécessaire
    # mask_uint8 = mask.astype(np.uint8) * 255
    
    # Calculer l'aire
    # area = _______
    
    # Trouver les contours
    # contours, _ = cv2.findContours(_______, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculer le périmètre
    # perimeter = 0
    # if len(contours) > 0:
    #     perimeter = cv2.arcLength(_______, True)
    
    # Calculer la bounding box
    # bbox = [0, 0, 0, 0]
    # if len(contours) > 0:
    #     x, y, w, h = cv2.boundingRect(_______)
    #     bbox = [x, y, w, h]
    
    # Calculer le centroïde
    # centroid = [0, 0]
    # if area > 0:
    #     moments = cv2.moments(mask_uint8)
    #     if moments['m00'] != 0:
    #         cx = int(moments['m10'] / moments['m00'])
    #         cy = int(moments['m01'] / moments['m00'])
    #         centroid = [cx, cy]
    
    # return {
    #     'area': area,
    #     'perimeter': perimeter,
    #     'bbox': bbox,
    #     'centroid': centroid
    # }
    pass


def compute_iou(mask1, mask2):
    """
    Calcule l'IoU (Intersection over Union) entre deux masques
    
    Args:
        mask1, mask2: Masques binaires
    
    Returns:
        float: Score IoU entre 0 et 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def mask_to_bbox(mask):
    """
    Extrait la bounding box d'un masque
    
    Args:
        mask: Masque binaire
    
    Returns:
        list: [x_min, y_min, x_max, y_max] ou None si masque vide
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def mask_area(mask: np.ndarray) -> int:
    """
    Aire en pixels (nombre de pixels True).
    """
    return int(np.count_nonzero(mask))


def mask_bbox(mask: np.ndarray):
    """
    BBox serrée du masque : (x1, y1, x2, y2).
    Si masque vide, retourner None.
    """
    if mask is None or mask.size == 0 or not mask.any():
        return None
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def mask_perimeter(mask: np.ndarray) -> float:
    """
    Périmètre approximatif via extraction de contours OpenCV.
    Si masque vide, retourner 0.0.
    """
    if mask is None or mask.size == 0 or not mask.any():
        return 0.0
    m = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    per = float(sum(cv2.arcLength(c, True) for c in contours))
    return per
