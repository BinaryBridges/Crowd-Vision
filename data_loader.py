import glob
import os
import pathlib

import cv2
import numpy as np

import config
from models import Person
from utils import normalize_l2

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _largest_face_embedding(app, img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Return the L2-normalized embedding for the largest detected face in an image.
    Uses df.normed_embedding if available, otherwise normalizes df.embedding.
    """
    dets = app.get(img_bgr)
    if not dets:
        return None

    # Pick largest bbox by area
    def area(df) -> float:
        x1, y1, x2, y2 = map(float, df.bbox)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    df = max(dets, key=area)

    emb = getattr(df, "normed_embedding", None)
    if emb is None:
        emb = getattr(df, "embedding", None)
        if emb is None:
            return None
        emb = normalize_l2(emb.astype(np.float32))
    else:
        emb = emb.astype(np.float32)

    return emb


def load_reference_identities(app, root: str) -> tuple[list[Person], list[str]]:
    """
    Scan known_faces/<category>/<person> folders and build a single list of Persons
    with a parallel list of their folder categories.

    Returns:
        people: List[Person]  (name, centroid, num_refs)
        categories: List[str] (same length as people; e.g., 'key', 'bad', 'vip', ...)

    """
    people: list[Person] = []
    categories: list[str] = []

    total_imgs = 0
    used_imgs = 0
    per_category_counts: dict[str, int] = {}
    per_person_counts: dict[tuple[str, str], int] = {}

    if not pathlib.Path(root).is_dir():
        print(f"[warn] KNOWN_DIR '{root}' does not exist or is not a directory.")
        return people, categories

    for category in sorted(os.listdir(root)):
        category_path = os.path.join(root, category)
        if not pathlib.Path(category_path).is_dir() or category.startswith("."):
            continue

        # Validate folder category against config.CATEGORY_META (name + color required)
        if not hasattr(config, "CATEGORY_META") or category not in config.CATEGORY_META:
            print(f"[warn] category '{category}' is not defined in config.CATEGORY_META (add a name & color to use it)")

        for person_name in sorted(os.listdir(category_path)):
            person_path = os.path.join(category_path, person_name)
            if not pathlib.Path(person_path).is_dir() or person_name.startswith("."):
                continue

            # Load all supported image files for this person
            img_files = []
            for ext in IMG_EXTS:
                img_files.extend(glob.glob(os.path.join(person_path, f"*{ext}")))
            img_files = sorted(img_files)
            total_imgs += len(img_files)

            embs = []
            for img_fp in img_files:
                img = cv2.imread(img_fp)
                if img is None:
                    continue
                emb = _largest_face_embedding(app, img)
                if emb is not None:
                    embs.append(emb)

            if embs:
                centroid = normalize_l2(np.stack(embs).astype(np.float32).mean(axis=0))
                people.append(Person(name=person_name, centroid=centroid, num_refs=len(embs), category=category))
                categories.append(category)
                used_imgs += len(embs)
                per_category_counts[category] = per_category_counts.get(category, 0) + 1
                per_person_counts[category, person_name] = len(embs)
                print(f"[loaded] [{category}] - {person_name}: {len(embs)} image(s)")
            else:
                print(f"[warn] no faces found for [{category}] - '{person_name}'")

    # Summary
    cats_summary = ", ".join(f"{c}:{n}" for c, n in sorted(per_category_counts.items())) if people else "none"

    print(f"[summary] people: {len(people)} ({cats_summary}); images scanned: {total_imgs}, images used: {used_imgs}")
    return people, categories
