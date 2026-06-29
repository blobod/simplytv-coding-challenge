"""Drop a handful of varied test images into images/ so you can try the flow.

Uses only the Python standard library (no extra installs). Downloads a few
real photos from Lorem Picsum. Re-running is safe: fixed seeds mean the same
images every time, and existing files are skipped.

Swap in your own photos any time — just put them in images/ and re-run index.py.
"""

import urllib.request
from pathlib import Path

IMAGES_DIR = Path("images")

# (filename, url) — distinct seeds give distinct real photos.
SAMPLES = [
    ("nature.jpg", "https://picsum.photos/seed/nature/800/600"),
    ("city.jpg", "https://picsum.photos/seed/city/800/600"),
    ("people.jpg", "https://picsum.photos/seed/people/800/600"),
    ("water.jpg", "https://picsum.photos/seed/water/800/600"),
    ("mountains.jpg", "https://picsum.photos/seed/mountains/800/600"),
    ("animal.jpg", "https://picsum.photos/seed/animal/800/600"),
    ("food.jpg", "https://picsum.photos/seed/food/800/600"),
    ("street.jpg", "https://picsum.photos/seed/street/800/600"),
    ("forest.jpg", "https://picsum.photos/seed/forest/800/600"),
    ("car.jpg", "https://picsum.photos/seed/car/800/600"),
    ("building.jpg", "https://picsum.photos/seed/building/800/600"),
    ("flower.jpg", "https://picsum.photos/seed/flower/800/600"),
]


def main():
    IMAGES_DIR.mkdir(exist_ok=True)
    for i, (name, url) in enumerate(SAMPLES, 1):
        dest = IMAGES_DIR / name
        if dest.exists():
            print(f"  [{i}/{len(SAMPLES)}] {name} already present, skipping")
            continue
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  [{i}/{len(SAMPLES)}] downloaded {name}")
        except Exception as e:
            print(f"  [{i}/{len(SAMPLES)}] failed {name}: {e}")

    print(f"\nDone. Images are in '{IMAGES_DIR}/'. Next: python index.py")


if __name__ == "__main__":
    main()
