"""
Real-time Face Recognition System using InsightFace and OpenCV

This script performs real-time face recognition using a webcam. It:
1. Loads known faces from a directory structure
2. Computes face embeddings using InsightFace's ArcFace model
3. Compares live webcam faces against known faces using cosine similarity
4. Displays bounding boxes and names for recognized faces

Usage: python face_id_webcam.py
Controls: Press 'q' to quit the application
"""

# Import required libraries
import os, glob, time, cv2, numpy as np # type: ignore
from dataclasses import dataclass
from typing import List
from insightface.app import FaceAnalysis # type: ignore  # Deep learning face analysis library

# --------- CONFIGURATION PARAMETERS ----------
KNOWN_DIR = "known_faces"   # Directory containing subdirectories named by person (e.g., "known_faces/john/", "known_faces/jane/")
SIM_THRESHOLD = 0.38        # Cosine similarity threshold for face matching (higher = stricter matching, range: 0-1)
MIN_FACE_SIZE = 64          # Minimum face size in pixels to process (filters out very small/distant faces)
SHOW_FPS = True             # Whether to display frames per second counter on screen
# ---------------------------------------------

@dataclass
class Person:
    """
    Data class to store information about a known person

    Attributes:
        name: The person's name (from directory name)
        centroid: Average face embedding vector representing this person (512-dimensional)
        num_refs: Number of reference images used to compute the centroid
    """
    name: str
    centroid: np.ndarray      # Shape: (512,) - normalized face embedding vector
    num_refs: int             # Number of images used to create this person's profile

def l2norm(v: np.ndarray) -> np.ndarray:
    """
    L2 normalize a vector or batch of vectors

    Args:
        v: Input vector(s) to normalize

    Returns:
        L2-normalized vector(s) with unit length

    Note: Adds small epsilon (1e-9) to prevent division by zero
    """
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9  # Compute L2 norm + epsilon
    return v / n  # Divide by norm to get unit vector

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between multiple vectors and a single vector

    Args:
        a: Matrix of vectors, shape (N, D) - multiple face embeddings
        b: Single vector, shape (D,) - query face embedding

    Returns:
        Array of cosine similarities, shape (N,) - one similarity score per vector in 'a'

    Note: Assumes both inputs are already L2-normalized, so cosine similarity = dot product
    """
    if a.size == 0: return np.array([])  # Handle empty input
    return (a @ b)  # Matrix-vector multiplication gives cosine similarities

def draw_label(img, text, x, y, scale=0.6, thickness=2):
    """
    Draw text with a black background rectangle for better visibility

    Args:
        img: Image to draw on (modified in-place)
        text: Text string to display
        x, y: Top-left position for the text
        scale: Font scale factor (default: 0.6)
        thickness: Text thickness in pixels (default: 2)
    """
    # Calculate text dimensions to size the background rectangle
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

    # Draw black background rectangle with padding
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 4), (0, 0, 0), -1)

    # Draw white text on top of the black background
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)

def load_known_people(app: FaceAnalysis, root: str) -> List[Person]:
    """
    Load known people from directory structure and compute their face embeddings

    Expected directory structure:
    known_faces/
    ├── person1/
    │   ├── photo1.jpg
    │   ├── photo2.png
    │   └── ...
    ├── person2/
    │   ├── photo1.jpg
    │   └── ...

    For each person:
    1. Finds all images in their subdirectory
    2. Extracts the largest face from each image
    3. Computes face embeddings using InsightFace
    4. Averages all embeddings to create a representative centroid

    Args:
        app: Initialized FaceAnalysis object for face detection and embedding
        root: Path to the known faces directory

    Returns:
        List of Person objects with computed centroids
    """
    people: List[Person] = []  # List to store all known people
    total_imgs, total_used = 0, 0  # Counters for statistics

    # Check if the known faces directory exists
    if not os.path.isdir(root):
        print(f"[WARN] No '{root}' directory found. Everyone will be 'Unknown'.")
        return people

    # Iterate through each person's subdirectory
    for name in sorted(os.listdir(root)):
        pdir = os.path.join(root, name)  # Full path to person's directory
        if not os.path.isdir(pdir): continue  # Skip files, only process directories

        embs = []  # List to store face embeddings for this person

        # Find all image files in the person's directory (jpg, jpeg, png)
        image_patterns = [glob.glob(os.path.join(pdir, ext)) for ext in ("*.jpg", "*.jpeg", "*.png")]
        image_paths = sorted(sum(image_patterns, []))  # Flatten and sort the list

        for img_path in image_paths:
            total_imgs += 1  # Count total images processed

            # Load the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"[skip] unreadable: {img_path}")
                continue

            # Detect faces in the image
            faces = app.get(img)  # Returns list of detected faces with embeddings
            if not faces:
                print(f"[skip] no faces detected: {img_path}")
                continue  # Skip images with no detected faces

            # Select the largest face (most prominent person in the image)
            # Face bbox format: [x1, y1, x2, y2]
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))

            # Extract the face embedding (already L2-normalized by InsightFace)
            emb = f.normed_embedding.astype(np.float32)  # Shape: (512,)
            embs.append(emb)
            total_used += 1  # Count faces actually used

        # If we found at least one face for this person
        if embs:
            # Stack all embeddings and compute the average (centroid)
            arr = np.stack(embs).astype(np.float32)  # Shape: (num_images, 512)
            centroid = l2norm(arr.mean(axis=0))  # Average and re-normalize

            # Create Person object and add to list
            people.append(Person(name=name, centroid=centroid, num_refs=len(embs)))
            print(f"[loaded] {name}: {len(embs)} image(s)")
        else:
            print(f"[warn] no faces found for '{name}'")

    # Print summary statistics
    print(f"[summary] persons: {len(people)}, images scanned: {total_imgs}, faces used: {total_used}")
    return people

def main():
    """
    Main function that runs the real-time face recognition system

    Process:
    1. Initialize InsightFace model for face detection and embedding
    2. Load known people from the directory structure
    3. Start webcam capture
    4. For each frame:
       - Detect faces
       - Extract embeddings
       - Compare against known people
       - Draw bounding boxes and labels
    5. Display results and handle user input
    """
    # Display ASCII art banner

    print("██████╗░░█████╗░██████╗░░█████╗░██╗░░██╗  ███████╗██╗░░░██╗███████╗")
    print("██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║░░██║  ██╔════╝╚██╗░██╔╝██╔════╝")
    print("██████╔╝██║░░██║██████╔╝██║░░╚═╝███████║  █████╗░░░╚████╔╝░█████╗░░")
    print("██╔═══╝░██║░░██║██╔══██╗██║░░██╗██╔══██║  ██╔══╝░░░░╚██╔╝░░██╔══╝░░")
    print("██║░░░░░╚█████╔╝██║░░██║╚█████╔╝██║░░██║  ███████╗░░░██║░░░███████╗")
    print("╚═╝░░░░░░╚════╝░╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝  ╚══════╝░░░╚═╝░░░╚══════╝")
    print("")

    # Initialize the face analysis model
    # "buffalo_l" is a pre-trained model that includes face detection and recognition
    # ctx_id=-1 uses CPU, ctx_id=0 would use GPU if available
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))  # det_size affects detection accuracy vs speed

    # Load all known people and prepare data structures for fast lookup
    people = load_known_people(app, KNOWN_DIR)
    names = np.array([p.name for p in people], dtype=object)  # Array of names for indexing

    # Stack all centroids into a matrix for vectorized similarity computation
    # If no people loaded, create empty array with correct shape (0, 512)
    print("Processing centroids...")
    centroids = np.stack([p.centroid for p in people]).astype(np.float32) if people else np.empty((0,512), np.float32)
    print("Done.")

    # Initialize webcam capture (index 0 = default camera)
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try a different index.")
    print("Done.")

    # Initialize FPS calculation variables
    fps, t_prev = 0.0, time.time()
    print("")
    
    print("Press 'q' to quit.")

    # Main processing loop
    while True:
        # Capture frame from webcam
        ok, frame = cap.read()
        if not ok:
            break  # Exit if camera disconnected or error

        # Detect all faces in the current frame
        faces = app.get(frame)  # Returns list of Face objects with bbox and embedding

        # Process each detected face
        for f in faces:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, f.bbox)  # Convert to integers for drawing
            w, h = x2 - x1, y2 - y1  # Calculate width and height

            # Skip very small faces (likely false positives or too distant)
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            # Extract face embedding (512-dimensional vector, already L2-normalized)
            emb = f.normed_embedding.astype(np.float32)

            # Default values for unknown faces
            label = "Unknown"
            score_txt = ""
            color = (0, 0, 255)  # Red color (BGR format) for unknown faces

            # If we have known people to compare against
            if centroids.size:
                # Compute cosine similarity with all known people at once
                sims = cosine_sim(centroids, emb)  # Shape: (num_people,)

                # Find the best match
                best_idx = int(np.argmax(sims))  # Index of most similar person
                best_sim = float(sims[best_idx])  # Similarity score (0-1)

                # Check if similarity exceeds threshold
                if best_sim >= SIM_THRESHOLD:
                    label = names[best_idx]  # Use the person's name
                    color = (0, 200, 0)  # Green color for recognized faces
                    score_txt = f" {best_sim:.2f}"  # Show confidence score

            # Draw bounding box around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with person's name and confidence score
            draw_label(frame, f"{label}{score_txt}", x1, y1)

        # Calculate and display FPS if enabled
        if SHOW_FPS:
            now = time.time()
            # Calculate instantaneous FPS
            inst = 1.0 / max(1e-6, (now - t_prev))  # Prevent division by zero
            # Apply exponential smoothing for stable FPS display
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            t_prev = now
            # Display FPS in top-left corner
            draw_label(frame, f"{fps:.1f} FPS", 10, 30)

        # Display the frame with annotations
        cv2.imshow("Face ID (webcam)", frame)

        # Check for 'q' key press to quit (waitKey returns key code)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup: release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """
    Script entry point - only run main() if this file is executed directly
    (not imported as a module)
    """
    main()
