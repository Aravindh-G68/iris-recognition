"""
Project  : IRIS RECOGNITION for Eye Detection on Humans for Security Purpose
Author   : Aravindh G
Duration : Three Months
Tech     : Python, OpenCV, scikit-learn

Description:
    The iris of a human eye contains characteristics unique to each individual.
    A camera captures the iris pattern, encodes it using image processing
    techniques, and verifies identity by comparing the extracted iris code
    against a pre-registered database. This system implements iris
    segmentation, feature extraction, and matching using SVM classification.
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# STEP 1 : Iris Segmentation
# ─────────────────────────────────────────────

def segment_iris(image):
    """
    Segment the iris region from an eye image:
    1. Grayscale conversion
    2. Gaussian blur to reduce noise
    3. Hough Circle Transform to detect pupil (inner boundary)
       and iris (outer boundary)
    4. Isolate and return the iris band
    """
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles (iris outer boundary)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=30,
        maxRadius=90
    )

    h, w = gray.shape
    cx, cy, r = w // 2, h // 2, min(h, w) // 3    # fallback center

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        cx, cy, r = circles[0]

    # Create circular mask for iris region
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    # Remove pupil area (inner 30%)
    cv2.circle(mask, (cx, cy), int(r * 0.3), 0, -1)

    iris_region = cv2.bitwise_and(gray, gray, mask=mask)

    # Crop and resize to fixed size
    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)
    x2 = min(cx + r, w)
    y2 = min(cy + r, h)
    iris_crop = iris_region[y1:y2, x1:x2]
    iris_resized = cv2.resize(iris_crop, (64, 64))
    return iris_resized, (cx, cy, r)


# ─────────────────────────────────────────────
# STEP 2 : Iris Code Encoding (IriCore-style)
# ─────────────────────────────────────────────

def encode_iris(iris_img):
    """
    Encode iris into a feature vector (IriCore-style encoding):
    1. Apply Gabor filter bank to extract texture features
    2. Flatten and normalise — this is the 'iris code'
    """
    features = []

    # Gabor filter bank at multiple orientations and scales
    for theta in [0, 45, 90, 135]:
        for sigma in [1, 2]:
            kernel = cv2.getGaborKernel(
                (21, 21),
                sigma=sigma,
                theta=np.deg2rad(theta),
                lambd=10,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(iris_img.astype(np.float32), -1, kernel)
            features.append(filtered.mean())
            features.append(filtered.std())

    # LBP-style texture histogram (simplified)
    lbp = np.zeros_like(iris_img, dtype=np.uint8)
    for i in range(1, iris_img.shape[0] - 1):
        for j in range(1, iris_img.shape[1] - 1):
            center  = iris_img[i, j]
            pattern = 0
            neighbors = [
                iris_img[i-1, j-1], iris_img[i-1, j], iris_img[i-1, j+1],
                iris_img[i,   j+1], iris_img[i+1, j+1], iris_img[i+1, j],
                iris_img[i+1, j-1], iris_img[i,   j-1]
            ]
            for k, nb in enumerate(neighbors):
                if nb >= center:
                    pattern |= (1 << k)
            lbp[i, j] = pattern

    hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
    features.extend(hist.tolist())

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────
# STEP 3 : Synthetic Eye Dataset
#          (Replace with real iris dataset path
#           e.g. CASIA or MMU iris database)
# ─────────────────────────────────────────────

def generate_eye_image(person_id, sample_id, noise_level=15):
    """
    Generate a synthetic eye image with a unique iris texture per person.
    Each person has a consistent ring texture + slight variation per sample.
    """
    rng = np.random.default_rng(person_id * 100 + sample_id)
    img = np.ones((150, 150, 3), dtype=np.uint8) * 220

    # Sclera (white area)
    cv2.ellipse(img, (75, 75), (70, 45), 0, 0, 360, (240, 240, 240), -1)

    # Iris base color (unique per person)
    base_hue = (person_id * 37) % 180
    iris_color = tuple(int(x) for x in cv2.cvtColor(
        np.uint8([[[base_hue, 180, 150]]]), cv2.COLOR_HSV2BGR)[0][0])
    cv2.circle(img, (75, 75), 42, iris_color, -1)

    # Iris texture (unique pattern per person)
    for _ in range(20):
        angle = rng.uniform(0, 360)
        r     = rng.integers(15, 38)
        x     = int(75 + r * np.cos(np.deg2rad(angle)))
        y     = int(75 + r * np.sin(np.deg2rad(angle)))
        c     = tuple(int(v) for v in
                      (np.array(iris_color) * rng.uniform(0.6, 1.1)).clip(0, 255))
        cv2.circle(img, (x, y), rng.integers(2, 6), c, -1)

    # Pupil
    cv2.circle(img, (75, 75), 18, (10, 10, 10), -1)
    cv2.circle(img, (79, 70),  4, (255, 255, 255), -1)   # reflection

    # Slight noise per sample
    noise = rng.integers(-noise_level, noise_level, img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def load_iris_dataset(database_path=None, n_persons=10, samples_per_person=20):
    """
    Load from real database OR generate synthetic data.
    Folder structure expected:
        database_path/
            person_001/  ← iris images (.jpg / .png)
            person_002/
            ...
    """
    features, labels = [], []

    if database_path and os.path.isdir(database_path):
        for label, person_dir in enumerate(sorted(os.listdir(database_path))):
            full_path = os.path.join(database_path, person_dir)
            if not os.path.isdir(full_path):
                continue
            for fname in os.listdir(full_path):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img = cv2.imread(os.path.join(full_path, fname))
                if img is None:
                    continue
                iris, _ = segment_iris(img)
                feat    = encode_iris(iris)
                features.append(feat)
                labels.append(label)
        if features:
            print(f"Loaded {len(features)} iris images from '{database_path}'.")
            return np.array(features), np.array(labels)

    # Synthetic fallback
    print("No database path provided — using synthetic iris dataset for demo.")
    for person_id in range(n_persons):
        for sample_id in range(samples_per_person):
            img  = generate_eye_image(person_id, sample_id)
            iris, _ = segment_iris(img)
            feat = encode_iris(iris)
            features.append(feat)
            labels.append(person_id)
    return np.array(features), np.array(labels)


# ─────────────────────────────────────────────
# STEP 4 : Train SVM Classifier
# ─────────────────────────────────────────────

def train_iris_svm(X_train, y_train):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf      = SVC(kernel='rbf', C=10, gamma='scale',
                   probability=True, random_state=42)
    clf.fit(X_scaled, y_train)
    return clf, scaler


# ─────────────────────────────────────────────
# STEP 5 : Identity Verification
# ─────────────────────────────────────────────

def verify_identity(clf, scaler, probe_image, claimed_id, threshold=0.5):
    """
    Verify if the probe iris image matches the claimed identity.
    Returns: (is_verified, confidence)
    """
    iris, _ = segment_iris(probe_image)
    feat    = encode_iris(iris).reshape(1, -1)
    feat_sc = scaler.transform(feat)
    probs   = clf.predict_proba(feat_sc)[0]
    pred_id = clf.predict(feat_sc)[0]

    if pred_id == claimed_id:
        confidence = probs[claimed_id] if claimed_id < len(probs) else max(probs)
        return True, confidence
    return False, probs[pred_id] if pred_id < len(probs) else max(probs)


# ─────────────────────────────────────────────
# STEP 6 : Visualise
# ─────────────────────────────────────────────

def plot_iris_samples(n_persons=5):
    """Show sample iris images and their segmented form."""
    fig, axes = plt.subplots(2, n_persons, figsize=(14, 5))
    fig.suptitle("Iris Recognition — Sample Images & Segmentation",
                 fontsize=13, fontweight='bold')
    for i in range(n_persons):
        img  = generate_eye_image(i, 0)
        iris, (cx, cy, r) = segment_iris(img)
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(overlay, (cx, cy), int(r * 0.3), (0, 0, 255), 2)
        axes[0, i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Person {i+1}", fontsize=9)
        axes[0, i].axis('off')
        axes[1, i].imshow(iris, cmap='gray')
        axes[1, i].set_title("Iris Code", fontsize=9)
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig('iris_samples.png', dpi=150)
    plt.show()
    print("Iris samples saved → iris_samples.png")


def plot_verification_result(probe_img, person_id, is_verified, confidence):
    """Display the verification result for a probe image."""
    iris, (cx, cy, r) = segment_iris(probe_img)
    overlay = probe_img.copy()
    color   = (0, 255, 0) if is_verified else (0, 0, 255)
    cv2.circle(overlay, (cx, cy), r, color, 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("Iris Identity Verification", fontsize=13, fontweight='bold')
    ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax1.set_title("Probe Eye (green=iris boundary)", fontsize=9)
    ax1.axis('off')
    ax2.imshow(iris, cmap='gray')
    status = "VERIFIED ✓" if is_verified else "REJECTED ✗"
    ax2.set_title(f"Person {person_id+1} → {status}\nConfidence: {confidence*100:.1f}%",
                  fontsize=9, color='green' if is_verified else 'red')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig('iris_verification_result.png', dpi=150)
    plt.show()
    print("Verification result saved → iris_verification_result.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  IRIS RECOGNITION SYSTEM")
    print("  Eye Detection for Human Security Identification")
    print("=" * 55)

    # Visualise samples
    plot_iris_samples(n_persons=5)

    # Load dataset
    # To use real data: load_iris_dataset(database_path='path/to/iris/db')
    X, y = load_iris_dataset(n_persons=10, samples_per_person=20)
    print(f"\nDataset  : {len(X)} iris samples | {len(set(y))} persons")
    print(f"Feature vector size: {X.shape[1]}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training : {len(X_train)} | Testing: {len(X_test)}")

    # Train SVM
    print("\nTraining Iris SVM classifier...")
    clf, scaler = train_iris_svm(X_train, y_train)
    print("Training complete.")

    # Evaluate
    X_test_sc = scaler.transform(X_test)
    y_pred    = clf.predict(X_test_sc)
    acc       = accuracy_score(y_test, y_pred)
    print("\n========== IRIS RECOGNITION RESULTS ==========")
    print(f"Recognition Accuracy : {acc * 100:.2f}%")
    print("\nClassification Report:")
    target_names = [f"Person {i+1}" for i in range(len(set(y)))]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Live verification demo
    print("\n--- Identity Verification Demo ---")
    for person_id in [0, 2, 4]:
        probe = generate_eye_image(person_id, sample_id=99, noise_level=20)
        verified, confidence = verify_identity(clf, scaler, probe,
                                               claimed_id=person_id)
        status = "VERIFIED" if verified else "REJECTED"
        print(f"Person {person_id+1} probe → {status} | Confidence: {confidence*100:.1f}%")

    # Plot one verification result
    probe_img = generate_eye_image(0, sample_id=99, noise_level=20)
    is_v, conf = verify_identity(clf, scaler, probe_img, claimed_id=0)
    plot_verification_result(probe_img, 0, is_v, conf)


if __name__ == '__main__':
    main()
