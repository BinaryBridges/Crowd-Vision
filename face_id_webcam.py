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
from typing import List, Tuple, Dict
from insightface.app import FaceAnalysis # type: ignore  # Deep learning face analysis library
from enum import Enum

class Status(Enum):
    CONFIRMED_KEY = 1
    CONFIRMED_BAD = 2
    TENTATIVE = 3
    LOST = 4


# --------- CONFIGURATION PARAMETERS ----------
KNOWN_DIR = "known_faces"   # Directory containing subdirectories named by person (e.g., "known_faces/john/", "known_faces/jane/")
SIM_THRESHOLD = 0.38        # Cosine similarity threshold for face matching (higher = stricter matching, range: 0-1)
MIN_FACE_SIZE = 32          # Minimum face size in pixels to process (filters out very small/distant faces)
SHOW_FPS = True             # Whether to display frames per second counter on screen
K = 10                      # Number of frames to keep track of for each face
MIN_HITS = 10                # Minimum number of hits before a face is confirmed
MAX_AGE = 5                 # Maximum number of frames to keep track of a face that has been lost
VERIFICATION_INTERVAL = 10   # Re-verify confirmed faces every N frames
NOTIFICATION_COOLDOWN = 100  # Frames to wait before allowing another notification for the same person
DETECTION_VECTOR_SIZE = 512  # Size of face embedding vectors when detecting new faces
VERIFICATION_VECTOR_SIZE = 128  # Size of face embedding vectors when verifying known faces
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

@dataclass
class TrackedFace:
    """
    Data class to store information about a face being tracked across frames

    Attributes:
        face_id: Unique identifier for this face
        bbox: Current bounding box coordinates [x1, y1, x2, y2]
        label: Recognized name or "Unknown"
        confidence: Similarity score (0-1) if recognized, 0 if unknown
        status: Current tracking status (CONFIRMED_KEY, TENTATIVE, LOST)
        hit_count: Number of consecutive frames this face has been detected
        age: Number of frames since this face was last detected
        embedding_full: Full 512-dimensional face embedding for recognition
        embedding_tracking: Reduced 128-dimensional embedding for fast tracking consistency
        frames_since_verification: Number of frames since last embedding verification
    """
    face_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    label: str
    confidence: float
    status: Status
    hit_count: int = 0  # Number of consecutive detections
    age: int = 0        # Frames since last detection
    embedding_full: np.ndarray = None  # Full 512-dim embedding for recognition
    embedding_tracking: np.ndarray = None  # Reduced 128-dim embedding for tracking
    frames_since_verification: int = 0  # Frames since last embedding check

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

def create_tracking_embedding(full_embedding: np.ndarray) -> np.ndarray:
    """
    Create a reduced 128-dimensional embedding for fast tracking consistency checks

    Args:
        full_embedding: Full 512-dimensional face embedding

    Returns:
        Reduced 128-dimensional embedding for tracking

    Note: Uses the first 128 dimensions which typically contain the most discriminative features
    """
    # Take the first 128 dimensions and re-normalize
    reduced = full_embedding[:VERIFICATION_VECTOR_SIZE]
    return l2norm(reduced)

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

def get_face_color(face: TrackedFace) -> Tuple[int, int, int]:
    """
    Get the appropriate color for a tracked face based on its status

    Args:
        face: TrackedFace object

    Returns:
        BGR color tuple
    """
    if face.status == Status.CONFIRMED_KEY:
        return (0, 200, 0) if face.label != "Unknown" else (255, 0, 0)  # Green for known, blue for unknown
    elif face.status == Status.CONFIRMED_BAD:
        return (0, 0, 255)  # Red for confirmed bad
    elif face.status == Status.TENTATIVE:
        return (0, 255, 255)  # Yellow for tentative
    else:  # LOST
        return (128, 128, 128)  # Gray for lost

def render_tracked_faces(frame, tracked_faces: List[TrackedFace]):
    """
    Render all tracked faces on the frame

    Args:
        frame: Image to draw on (modified in-place)
        tracked_faces: List of TrackedFace objects to render
    """
    for face in tracked_faces:
        # Skip rendering lost faces
        if face.status == Status.LOST:
            continue

        x1, y1, x2, y2 = face.bbox
        color = get_face_color(face)

        # Draw bounding box around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Create label text with confidence score and status
        status_text = face.status.name.lower()
        if face.confidence > 0:
            label_text = f"{face.label} {face.confidence:.2f} ({status_text})"
        else:
            label_text = f"{face.label} ({status_text})"

        # Draw label with person's name, confidence score, and status
        draw_label(frame, label_text, x1, y1)

def calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        bbox1, bbox2: Bounding boxes in format (x1, y1, x2, y2)

    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def update_tracked_faces(tracked_faces: List[TrackedFace], detected_faces: List, app: FaceAnalysis,
                        names: np.ndarray, centroids: np.ndarray, person_types: np.ndarray, face_id_counter: int,
                        notified_faces: Dict[str, int], frame_count: int) -> Tuple[List[TrackedFace], int]:
    """
    Update tracked faces with new detections using IoU matching
    Only compute embeddings for new and tentative faces, use IoU for confirmed faces

    Args:
        tracked_faces: List of currently tracked faces
        detected_faces: List of Face objects from InsightFace detection
        app: FaceAnalysis object for embedding computation
        names: Array of known person names
        centroids: Array of known person embeddings
        face_id_counter: Current face ID counter

    Returns:
        Updated list of tracked faces and new face_id_counter
    """
    # Age all existing tracked faces and increment verification counters
    for face in tracked_faces:
        face.age += 1
        if face.status == Status.CONFIRMED_KEY or face.status == Status.CONFIRMED_BAD:
            face.frames_since_verification += 1
        if face.age > MAX_AGE:
            face.status = Status.LOST

    # Convert detected faces to bboxes for processing
    detection_bboxes = []
    for f in detected_faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        w, h = x2 - x1, y2 - y1
        if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
            detection_bboxes.append((x1, y1, x2, y2))

    # First, try to match detections using embeddings for better accuracy
    matched_detections = set()
    matched_faces = set()

    # For each detection, find the best match using both IoU and embedding similarity
    for detection_idx, bbox in enumerate(detection_bboxes):
        detected_face = detected_faces[detection_idx]
        embedding = detected_face.normed_embedding.astype(np.float32)

        best_match = None
        best_score = 0.0

        # Find best matching tracked face using combined IoU and embedding similarity
        for face in tracked_faces:
            if face.status == Status.LOST or face.face_id in matched_faces:
                continue

            # Calculate IoU score
            iou = calculate_iou(bbox, face.bbox)

            # Calculate embedding similarity if available (use tracking embedding for speed)
            embedding_sim = 0.0
            if face.embedding_tracking is not None:
                tracking_emb = create_tracking_embedding(embedding)
                embedding_sim = cosine_sim(face.embedding_tracking.reshape(1, -1), tracking_emb)[0]

            # Combined score: prioritize embedding similarity for confirmed faces, IoU for tentative
            if face.status == Status.CONFIRMED_KEY or face.status == Status.CONFIRMED_BAD:
                # For confirmed faces, require high embedding similarity OR very high IoU
                if embedding_sim > 0.7 or iou > 0.7:
                    score = embedding_sim * 0.7 + iou * 0.3
                else:
                    score = 0.0  # Not a match
            else:
                # For tentative faces, use IoU primarily but consider embedding
                if iou > 0.3:
                    score = iou * 0.7 + embedding_sim * 0.3
                else:
                    score = 0.0

            if score > best_score and score > 0.4:  # Minimum threshold for matching
                best_score = score
                best_match = face

        if best_match:
            # Update existing tracked face
            best_match.bbox = bbox
            best_match.age = 0
            best_match.hit_count += 1

            # Handle embedding computation based on status
            if best_match.status == Status.TENTATIVE:
                # For tentative faces, always check against ALL known faces to verify identity
                # This prevents misidentification of similar-looking people

                # Perform fresh face recognition against all known faces
                new_label = "Unknown"
                new_confidence = 0.0

                if centroids.size > 0:
                    sims = cosine_sim(centroids, embedding)
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])

                    if best_sim >= SIM_THRESHOLD:
                        new_label = names[best_idx]
                        new_confidence = best_sim
                        person_type = person_types[best_idx]

                # Check if the identity has changed from what we initially thought
                if best_match.label != new_label:
                    # Identity changed! Reset the tracking for this face
                    print(f"[IDENTITY CHANGED] Face ID {best_match.face_id}: {best_match.label} -> {new_label} (confidence: {new_confidence:.3f})")
                    best_match.hit_count = 1  # Reset to step 1
                    best_match.label = new_label
                    best_match.confidence = new_confidence
                    best_match.embedding_full = embedding
                    best_match.embedding_tracking = create_tracking_embedding(embedding)
                else:
                    # Same identity, but verify tracking embedding consistency (using 128-dim for speed)
                    if best_match.embedding_tracking is not None:
                        current_tracking_emb = create_tracking_embedding(embedding)
                        embedding_similarity = cosine_sim(best_match.embedding_tracking.reshape(1, -1), current_tracking_emb)[0]
                        if embedding_similarity < 0.6:  # Lower threshold for same person verification
                            # Embedding changed significantly - might be different person or lighting change
                            print(f"[EMBEDDING DRIFT] Face ID {best_match.face_id} ({best_match.label}): tracking similarity {embedding_similarity:.3f} - resetting")
                            best_match.hit_count = 1  # Reset to step 1

                    # Update with new data
                    best_match.label = new_label
                    best_match.confidence = new_confidence
                    best_match.embedding_full = embedding
                    best_match.embedding_tracking = create_tracking_embedding(embedding)

                # Check if face should be confirmed (only if we have enough consistent hits)
                if best_match.hit_count >= MIN_HITS:
                    # Determine person type and set appropriate status
                    if best_match.label != "Unknown" and centroids.size > 0:
                        # Find the person type from our arrays
                        person_idx = np.where(names == best_match.label)[0]
                        if len(person_idx) > 0:
                            person_type = person_types[person_idx[0]]
                            if person_type == 'key':
                                best_match.status = Status.CONFIRMED_KEY
                                status_text = "CONFIRMED_KEY"
                                notification_prefix = "🔑 KEY PERSON"
                            else:  # person_type == 'bad'
                                best_match.status = Status.CONFIRMED_BAD
                                status_text = "CONFIRMED_BAD"
                                notification_prefix = "⚠️ BAD PERSON"
                        else:
                            # Person not found in arrays - this shouldn't happen for known people
                            # Log this as a potential data loading issue
                            print(f"[WARNING] Person '{best_match.label}' recognized but not found in person_types array!")
                            print(f"[DEBUG] Available names: {list(names)}")
                            # Default to tentative status until we can verify
                            best_match.status = Status.TENTATIVE
                            status_text = "TENTATIVE_UNVERIFIED"
                            notification_prefix = "❓ UNVERIFIED"
                    else:
                        # Unknown person
                        best_match.status = Status.CONFIRMED_KEY  # Default for unknown
                        status_text = "CONFIRMED_UNKNOWN"
                        notification_prefix = "❓ UNKNOWN PERSON"

                    best_match.frames_since_verification = 0  # Reset verification counter
                    print(f"[{status_text}] Face ID {best_match.face_id} ({best_match.label}) after {best_match.hit_count} consistent detections")

                    # Send notification if this person hasn't been notified recently
                    person_key = best_match.label

                    if person_key not in notified_faces or (frame_count - notified_faces[person_key]) >= NOTIFICATION_COOLDOWN:
                        if best_match.label == "Unknown":
                            print(f"[NOTIFICATION] {notification_prefix} detected and confirmed")
                        else:
                            print(f"[NOTIFICATION] {notification_prefix} - {best_match.label} detected and confirmed")
                        notified_faces[person_key] = frame_count

            elif best_match.status == Status.CONFIRMED_KEY or best_match.status == Status.CONFIRMED_BAD:
                # For confirmed faces, only do periodic verification
                if best_match.frames_since_verification >= VERIFICATION_INTERVAL:
                    # Time for periodic verification - check if it's still the right person
                    print(f"[PERIODIC CHECK] Verifying Face ID {best_match.face_id} ({best_match.label}) after {best_match.frames_since_verification} frames")

                    # Perform fresh face recognition against all known faces
                    verified_label = "Unknown"
                    verified_confidence = 0.0

                    if centroids.size > 0:
                        sims = cosine_sim(centroids, embedding)
                        best_idx = int(np.argmax(sims))
                        best_sim = float(sims[best_idx])

                        if best_sim >= SIM_THRESHOLD:
                            verified_label = names[best_idx]
                            verified_confidence = best_sim
                            verified_person_type = person_types[best_idx]

                    # Check if identity has changed
                    if best_match.label != verified_label:
                        # Identity changed! Demote back to tentative and reset
                        print(f"[IDENTITY MISMATCH] Face ID {best_match.face_id}: {best_match.label} -> {verified_label} - demoting to tentative")
                        best_match.status = Status.TENTATIVE
                        best_match.hit_count = 1
                        best_match.label = verified_label
                        best_match.confidence = verified_confidence
                        best_match.embedding = embedding
                        best_match.frames_since_verification = 0
                    else:
                        # Identity confirmed - update confidence, status if needed, and reset verification counter
                        if verified_label != "Unknown":
                            # Update status based on current person type
                            if verified_person_type == 'key' and best_match.status != Status.CONFIRMED_KEY:
                                best_match.status = Status.CONFIRMED_KEY
                                print(f"[STATUS UPDATE] Face ID {best_match.face_id} ({best_match.label}) -> CONFIRMED_KEY")
                            elif verified_person_type == 'bad' and best_match.status != Status.CONFIRMED_BAD:
                                best_match.status = Status.CONFIRMED_BAD
                                print(f"[STATUS UPDATE] Face ID {best_match.face_id} ({best_match.label}) -> CONFIRMED_BAD")

                        print(f"[VERIFIED] Face ID {best_match.face_id} ({best_match.label}) - confidence: {verified_confidence:.3f}")
                        best_match.confidence = verified_confidence
                        best_match.embedding_full = embedding
                        best_match.embedding_tracking = create_tracking_embedding(embedding)
                        best_match.frames_since_verification = 0
                else:
                    # No verification needed - just track with IoU (no embedding computation)
                    # This is the efficient path for confirmed faces
                    pass

            matched_detections.add(detection_idx)
            matched_faces.add(best_match.face_id)

    # Handle unmatched detections - check for duplicates and lost face recovery
    for detection_idx, bbox in enumerate(detection_bboxes):
        if detection_idx not in matched_detections:
            # Get embedding for new face
            detected_face = detected_faces[detection_idx]
            embedding = detected_face.normed_embedding.astype(np.float32)

            # Check if this embedding matches any existing tracked face (including LOST ones)
            best_existing_match = None
            best_embedding_similarity = 0.0

            for existing_face in tracked_faces:
                if existing_face.embedding_full is not None:
                    # Use full 512-dim embedding for accurate person matching
                    similarity = cosine_sim(existing_face.embedding_full.reshape(1, -1), embedding)[0]
                    if similarity > best_embedding_similarity and similarity > 0.75:  # High threshold for same person
                        best_embedding_similarity = similarity
                        best_existing_match = existing_face

            if best_existing_match:
                # This is likely the same person as an existing tracked face
                if best_existing_match.status == Status.LOST:
                    # Reactivate lost face - restart tentative process
                    best_existing_match.bbox = bbox
                    best_existing_match.age = 0
                    best_existing_match.hit_count = 1
                    best_existing_match.status = Status.TENTATIVE
                    best_existing_match.embedding_full = embedding
                    best_existing_match.embedding_tracking = create_tracking_embedding(embedding)
                    best_existing_match.frames_since_verification = 0

                    # Re-do face recognition with new embedding
                    label = "Unknown"
                    confidence = 0.0

                    if centroids.size > 0:
                        sims = cosine_sim(centroids, embedding)
                        best_idx = int(np.argmax(sims))
                        best_sim = float(sims[best_idx])

                        if best_sim >= SIM_THRESHOLD:
                            label = names[best_idx]
                            confidence = best_sim

                    best_existing_match.label = label
                    best_existing_match.confidence = confidence

                    print(f"[REACTIVATED] Face ID {best_existing_match.face_id} ({label}) - similarity: {best_embedding_similarity:.3f}")
                else:
                    # This person is already being tracked - skip to prevent duplicates
                    print(f"[DUPLICATE PREVENTED] Skipping detection - person already tracked as ID {best_existing_match.face_id} ({best_existing_match.label}) - similarity: {best_embedding_similarity:.3f}")
                    continue
            else:
                # This is truly a new person
                # Perform face recognition for new face
                label = "Unknown"
                confidence = 0.0

                if centroids.size > 0:
                    sims = cosine_sim(centroids, embedding)
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])

                    if best_sim >= SIM_THRESHOLD:
                        label = names[best_idx]
                        confidence = best_sim
                        person_type = person_types[best_idx]

                new_face = TrackedFace(
                    face_id=face_id_counter,
                    bbox=bbox,
                    label=label,
                    confidence=confidence,
                    status=Status.TENTATIVE,
                    hit_count=1,
                    age=0,
                    embedding_full=embedding,
                    embedding_tracking=create_tracking_embedding(embedding),
                    frames_since_verification=0
                )
                tracked_faces.append(new_face)
                print(f"[NEW FACE] ID {face_id_counter} ({label}) - confidence: {confidence:.3f}")
                face_id_counter += 1

    # Enforce "one box per name" rule - remove duplicates and prioritize confirmed faces
    name_to_faces = {}
    for face in tracked_faces:
        if face.status != Status.LOST:
            name = face.label
            if name not in name_to_faces:
                name_to_faces[name] = []
            name_to_faces[name].append(face)

    # For each name, keep only the best face (prioritize confirmed, then highest confidence)
    for name, faces_with_name in name_to_faces.items():
        if len(faces_with_name) > 1:
            # Sort by: 1) Status (CONFIRMED_KEY/BAD first), 2) Confidence, 3) Hit count
            faces_with_name.sort(key=lambda f: (
                f.status not in [Status.CONFIRMED_KEY, Status.CONFIRMED_BAD],  # False (0) for confirmed comes first
                -f.confidence,  # Higher confidence first
                -f.hit_count    # Higher hit count first
            ))

            # Keep the best one, mark others as LOST
            best_face = faces_with_name[0]
            for face in faces_with_name[1:]:
                face.status = Status.LOST
                face.age = MAX_AGE + 1  # Mark for immediate removal
                print(f"[DUPLICATE REMOVED] Face ID {face.face_id} ({face.label}) removed - keeping ID {best_face.face_id}")

    # Remove faces that have been lost for too long (but keep them longer for potential reactivation)
    # Only remove faces that have been LOST for more than MAX_AGE * 2 frames
    tracked_faces = [face for face in tracked_faces if not (face.status == Status.LOST and face.age > MAX_AGE * 2)]

    return tracked_faces, face_id_counter

def cleanup_old_notifications(notified_faces: Dict[str, int], current_frame: int):
    """
    Remove old notification entries to prevent memory buildup

    Args:
        notified_faces: Dictionary tracking when each person was last notified
        current_frame: Current frame number
    """
    # Remove entries older than the cooldown period
    expired_keys = [
        person for person, frame in notified_faces.items()
        if current_frame - frame > NOTIFICATION_COOLDOWN
    ]
    for key in expired_keys:
        del notified_faces[key]

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
    key_people: List[Person] = []  # List to store all known people with key access
    bad_people: List[Person] = []  # List to store all known people who are known threats
    total_imgs, key_faces, bad_faces = 0, 0, 0  # Counters for statistics

    # Check if the known faces directory exists
    if not os.path.isdir(root):
        print(f"[WARN] No '{root}' directory found. Everyone will be 'Unknown'.")
        return key_people, bad_people

    # Iterate through each person's subdirectory
    for major_folder in sorted(os.listdir(root)):
        major_path = os.path.join(root, major_folder)
        if not os.path.isdir(major_path):
            continue  # skip if there is a file

        person_type = major_folder
    
        # nested loop for subdirectories inside major_folder
        for name in sorted(os.listdir(major_path)):
            pdir = os.path.join(major_path, name)
            if os.path.isdir(pdir):
                print(f"Subdir: {pdir}")

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
                    continue  # Skip images with no detected faces

                # Select the largest face (most prominent person in the image)
                # Face bbox format: [x1, y1, x2, y2]
                f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))

                # Extract the face embedding (already L2-normalized by InsightFace)
                emb = f.normed_embedding.astype(np.float32)  # Shape: (DETECTION_VECTOR_SIZE,)
                embs.append(emb)

            # If we found at least one face for this person
            if embs:
                # Stack all embeddings and compute the average (centroid)
                arr = np.stack(embs).astype(np.float32)  # Shape: (num_images, DETECTION_VECTOR_SIZE)
                centroid = l2norm(arr.mean(axis=0))  # Average and re-normalize

                # Create Person object and add to list
                if(person_type == "key"):
                    key_people.append(Person(name=name, centroid=centroid, num_refs=len(embs)))
                    key_faces += len(embs)
                elif(person_type == "bad"):
                    bad_people.append(Person(name=name, centroid=centroid, num_refs=len(embs)))
                    bad_faces += len(embs)
                print(f"[loaded] [{person_type}] - {name}: {len(embs)} image(s)")
            else:
                print(f"[warn] no faces found for [{person_type}] - '{name}'")

    # Print summary statistics
    print(f"[summary] key faces: {len(key_people)}, bad people: {len(bad_people)}, images scanned: {total_imgs}, key faces used: {key_faces}, bad faces: {bad_faces}")
    return key_people, bad_people

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
    key_people, bad_people = load_known_people(app, KNOWN_DIR)

    # Combine all people for unified processing while maintaining type information
    all_people = key_people + bad_people

    # Create arrays for fast vectorized lookup
    names = np.array([p.name for p in all_people], dtype=object)  # Array of names for indexing
    person_types = np.array(['key'] * len(key_people) + ['bad'] * len(bad_people), dtype=object)  # Track person types

    # Stack all centroids into a matrix for vectorized similarity computation
    # If no people loaded, create empty array with correct shape (0, DETECTION_VECTOR_SIZE)
    centroids = np.stack([p.centroid for p in all_people]).astype(np.float32) if all_people else np.empty((0,DETECTION_VECTOR_SIZE), np.float32)

    # Print summary of loaded people
    print(f"[DATA STRUCTURES] Loaded {len(key_people)} key people, {len(bad_people)} bad people")
    print(f"[DATA STRUCTURES] Combined arrays: {len(names)} names, {len(person_types)} types, {centroids.shape[0]} centroids")
    print(f"[DEBUG] Key people names: {[p.name for p in key_people]}")
    print(f"[DEBUG] Bad people names: {[p.name for p in bad_people]}")
    print(f"[DEBUG] Names array: {list(names)}")
    print(f"[DEBUG] Person types array: {list(person_types)}")

    # Initialize webcam capture (index 0 = default camera)
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try a different index.")
    print("Done.")

    # Initialize FPS calculation variables, face ID counter, tracking list, and notification system
    fps, t_prev = 0.0, time.time()
    face_id_counter = 0  # Counter to assign unique IDs to detected faces
    tracked_faces: List[TrackedFace] = []  # List to maintain tracked faces across frames
    notified_faces: Dict[str, int] = {}  # Track when each person was last notified
    frame_count = 0  # Global frame counter for notification cooldown

    print("")
    
    print("Press 'q' to quit.")

    # Main processing loop
    while True:
        # Increment frame counter
        frame_count += 1

        # Capture frame from webcam
        ok, frame = cap.read()
        if not ok:
            break  # Exit if camera disconnected or error

        # Detect all faces in the current frame
        faces = app.get(frame)  # Returns list of Face objects with bbox and embedding

        # Update tracked faces with new detections and handle notifications
        tracked_faces, face_id_counter = update_tracked_faces(
            tracked_faces, faces, app, names, centroids, person_types, face_id_counter, notified_faces, frame_count
        )

        # Clean up old notification entries periodically (every 100 frames)
        if frame_count % 100 == 0:
            cleanup_old_notifications(notified_faces, frame_count)

        # Render all tracked faces
        render_tracked_faces(frame, tracked_faces)

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
