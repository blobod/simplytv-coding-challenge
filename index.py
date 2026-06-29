"""Offline indexing: embed every image once with CLIP, build thumbnails, save to disk.

Run this whenever the images/ folder changes. Search reads the outputs; it never
re-embeds at query time.

Outputs:
  embeddings.npy  - float32 matrix, L2-normalized, one row per image
  index.json      - filenames aligned to the rows of embeddings.npy
  thumbnails/     - one small JPEG per image, served as-is by the app
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

IMAGES_DIR = Path("images")
THUMBS_DIR = Path("thumbnails")
EMBEDDINGS_PATH = Path("embeddings.npy")
INDEX_PATH = Path("index.json")

MODEL_NAME = "clip-ViT-B-32"
THUMB_SIZE = (256, 256)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def find_images(folder):
    """Return image paths sorted by name for a stable, reproducible index."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)


def make_thumbnail(image, dest):
    """Save a small RGB JPEG thumbnail (cheap to serve, no per-request work)."""
    thumb = image.convert("RGB")
    thumb.thumbnail(THUMB_SIZE)
    thumb.save(dest, "JPEG", quality=85)


def main():
    if not IMAGES_DIR.is_dir():
        raise SystemExit(f"No '{IMAGES_DIR}/' folder found. Add images and retry.")

    paths = find_images(IMAGES_DIR)
    if not paths:
        raise SystemExit(f"No images found in '{IMAGES_DIR}/'.")

    print(f"Found {len(paths)} images. Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    THUMBS_DIR.mkdir(exist_ok=True)

    images, filenames = [], []
    for i, path in enumerate(paths, 1):
        try:
            image = Image.open(path)
            image.load()
        except Exception as e:
            print(f"  [{i}/{len(paths)}] skipped {path.name}: {e}")
            continue

        make_thumbnail(image, THUMBS_DIR / f"{path.stem}.jpg")
        images.append(image.convert("RGB"))
        filenames.append(path.name)
        print(f"  [{i}/{len(paths)}] {path.name}")

    print("Embedding images with CLIP...")
    embeddings = model.encode(
        images,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity becomes a plain dot product
        show_progress_bar=True,
    ).astype("float32")

    np.save(EMBEDDINGS_PATH, embeddings)
    INDEX_PATH.write_text(json.dumps({"filenames": filenames}, indent=2))

    print(f"\nDone. {len(filenames)} images indexed.")
    print(f"  {EMBEDDINGS_PATH}  shape={embeddings.shape}")
    print(f"  {INDEX_PATH}")
    print(f"  {THUMBS_DIR}/")


if __name__ == "__main__":
    main()
