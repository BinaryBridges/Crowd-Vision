import glob
import os

import cv2
import numpy as np

from models import Person
from utils import normalize_l2


def load_reference_identities(app, root: str) -> tuple[list[Person], list[Person]]:
    """Scan known_faces/ folder, compute per-person centroids from the largest face in each image.
    Returns (key_people, bad_people) lists of Person.
    """
    key_people: list[Person] = []
    bad_people: list[Person] = []

    total_imgs = 0
    key_used = 0
    bad_used = 0

    if not os.path.isdir(root):
        print(f"[WARN] No '{root}' directory found. Everyone will be 'Unknown'.")
        return key_people, bad_people

    categories = sorted(os.listdir(root))
    for category in categories:
        category_path = os.path.join(root, category)
        if not os.path.isdir(category_path):
            continue

        for person_name in sorted(os.listdir(category_path)):
            person_dir = os.path.join(category_path, person_name)
            if not os.path.isdir(person_dir):
                continue

            print(f"Subdir: {person_dir}")
            embs = []

            pattern_lists = [
                glob.glob(os.path.join(person_dir, ext))
                for ext in ("*.jpg", "*.jpeg", "*.png")
            ]
            image_paths = sorted(sum(pattern_lists, []))

            for img_path in image_paths:
                total_imgs += 1
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[skip] unreadable: {img_path}")
                    continue
                faces = app.get(img)
                if not faces:
                    continue

                # Largest face in the image
                f = max(
                    faces,
                    key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                )
                emb = f.normed_embedding.astype(np.float32)
                embs.append(emb)

            if embs:
                arr = np.stack(embs).astype(np.float32)
                centroid = normalize_l2(arr.mean(axis=0))
                if category == "bad":
                    bad_people.append(
                        Person(name=person_name, centroid=centroid, num_refs=len(embs))
                    )
                    bad_used += len(embs)
                else:
                    key_people.append(
                        Person(name=person_name, centroid=centroid, num_refs=len(embs))
                    )
                    key_used += len(embs)
                print(f"[loaded] [{category}] - {person_name}: {len(embs)} image(s)")
            else:
                print(f"[warn] no faces found for [{category}] - '{person_name}'")

    print(
        f"[summary] key faces: {len(key_people)}, bad people: {len(bad_people)}, "
        f"images scanned: {total_imgs}, key faces used: {key_used}, bad faces: {bad_used}"
    )
    return key_people, bad_people
