# face_id_webcam.py
# Run: python face_id_webcam.py
# q = quit

import os, glob, time, cv2, numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from insightface.app import FaceAnalysis

# --------- CONFIG ----------
KNOWN_DIR = "known_faces"   # folder with subfolders named by person
SIM_THRESHOLD = 0.38        # ArcFace cosine similarity threshold (↑ stricter)
MIN_FACE_SIZE = 64          # ignore tiny detections (pixels)
SHOW_FPS = True
# ---------------------------

@dataclass
class Person:
    name: str
    centroid: np.ndarray      # (D,)
    num_refs: int

def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
    return v / n

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (N, D), b: (D,)
    if a.size == 0: return np.array([])
    return (a @ b)  # assume both already L2-normalized

def draw_label(img, text, x, y, scale=0.6, thickness=2):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)

def load_known_people(app: FaceAnalysis, root: str) -> List[Person]:
    """
    Walk KNOWN_DIR/<Name>/*.jpg, embed largest face per image,
    compute an L2-normalized centroid per person.
    """
    people: List[Person] = []
    total_imgs, total_used = 0, 0

    if not os.path.isdir(root):
        print(f"[WARN] No '{root}' directory found. Everyone will be 'Unknown'.")
        return people

    for name in sorted(os.listdir(root)):
        pdir = os.path.join(root, name)
        if not os.path.isdir(pdir): continue
        embs = []
        for img_path in sorted(sum([glob.glob(os.path.join(pdir, ext)) for ext in ("*.jpg", "*.jpeg", "*.png")], [])):
            total_imgs += 1
            img = cv2.imread(img_path)
            if img is None: 
                print(f"[skip] unreadable: {img_path}")
                continue
            faces = app.get(img)
            if not faces: 
                continue
            # largest face
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            emb = f.normed_embedding.astype(np.float32)  # already L2-normalized
            embs.append(emb)
            total_used += 1
        if embs:
            arr = np.stack(embs).astype(np.float32)
            centroid = l2norm(arr.mean(axis=0))  # normalize the mean
            people.append(Person(name=name, centroid=centroid, num_refs=len(embs)))
            print(f"[loaded] {name}: {len(embs)} image(s)")
        else:
            print(f"[warn] no faces found for '{name}'")

    print(f"[summary] persons: {len(people)}, images scanned: {total_imgs}, faces used: {total_used}")
    return people

def main():
    # Use CPU by default (ctx_id = -1). If you have a CUDA GPU, set ctx_id=0.
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    people = load_known_people(app, KNOWN_DIR)
    names = np.array([p.name for p in people], dtype=object)
    centroids = np.stack([p.centroid for p in people]).astype(np.float32) if people else np.empty((0,512), np.float32)

    cap = cv2.VideoCapture(0)  # laptop cam
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try a different index.")

    fps, t_prev = 0.0, time.time()
    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = app.get(frame)  # detects + gives embeddings
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            w, h = x2 - x1, y2 - y1
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            emb = f.normed_embedding.astype(np.float32)  # L2-normalized, shape (512,)
            label = "Unknown"
            score_txt = ""
            color = (0, 0, 255)  # red for unknown by default

            if centroids.size:
                sims = cosine_sim(centroids, emb)  # (N,)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                if best_sim >= SIM_THRESHOLD:
                    label = names[best_idx]
                    color = (0, 200, 0)  # green for known
                    score_txt = f" {best_sim:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, f"{label}{score_txt}", x1, y1)

        if SHOW_FPS:
            now = time.time()
            inst = 1.0 / max(1e-6, (now - t_prev))
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            t_prev = now
            draw_label(frame, f"{fps:.1f} FPS", 10, 30)

        cv2.imshow("Face ID (webcam)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
