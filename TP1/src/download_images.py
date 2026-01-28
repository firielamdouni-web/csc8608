"""
Télécharge un petit jeu d'images reproductible (via picsum.photos seeds)
et génère des vignettes pour le rapport.

Usage (depuis le repo root):
    python TP1/src/download_images.py --n 12

Le script sauvegarde les images dans `TP1/data/images/`
et les vignettes dans `TP1/report/img/`.
"""
import argparse
import os
import urllib.request
from pathlib import Path
from PIL import Image


DEFAULT_SEEDS = [
    "simple1", "simple2", "simple3",
    "charged1", "charged2", "charged3",
    "difficult1", "difficult2", "difficult3",
    "misc1", "misc2", "misc3"
]


def download_image(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, out_path)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def make_thumbnail(src_path, dst_path, size=(400, 300)):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im.thumbnail(size)
            im.save(dst_path, format="PNG")
        return True
    except Exception as e:
        print(f"Failed to create thumbnail for {src_path}: {e}")
        return False


def main(n=12, width=1024, height=768):
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "data" / "images"
    thumb_dir = repo_root / "report" / "img"

    seeds = DEFAULT_SEEDS[:n]

    downloaded = []
    for i, seed in enumerate(seeds, start=1):
        url = f"https://picsum.photos/seed/{seed}/{width}/{height}"
        fname = f"{i:02d}_{seed}.jpg"
        out_path = out_dir / fname
        ok = download_image(url, out_path)
        if ok:
            downloaded.append(fname)
            thumb_path = thumb_dir / f"thumb_{i:02d}.png"
            make_thumbnail(out_path, thumb_path)

    print(f"Downloaded {len(downloaded)} images into {out_dir}")
    print(f"Thumbnails saved into {thumb_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12, help="Number of images to download (max 12)")
    parser.add_argument("--w", type=int, default=1024, help="Width of downloaded images")
    parser.add_argument("--h", type=int, default=768, help="Height of downloaded images")
    args = parser.parse_args()
    n = max(1, min(args.n, len(DEFAULT_SEEDS)))
    main(n=n, width=args.w, height=args.h)
