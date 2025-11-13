from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Optional HEIC image support
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "signature.csv"
SIGNATURE_DIR = BASE_DIR / "Signature"
PHOTO_DIR = BASE_DIR / "Photos" / "Identity sized photo (File responses)"


# --- DATA STRUCTURE ---
@dataclass
class SignatureTemplate:
    name_key: str
    display_name: str
    path: Path
    image: np.ndarray
    keypoints: Optional[List[cv2.KeyPoint]] = None
    descriptors: Optional[np.ndarray] = None


# --- HELPER FUNCTIONS ---
def normalize_name(raw: str) -> str:
    """Normalize names to lowercase alphanumeric string."""
    return "".join(ch for ch in raw.lower() if ch.isalnum())


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load CSV and add normalized name key."""
    df = pd.read_csv(CSV_PATH).fillna("")
    df["NameKey"] = df["Name"].map(normalize_name)
    return df


def _load_image_as_gray(path: Path) -> Optional[np.ndarray]:
    """Load image as grayscale and resize for feature extraction."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None and path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        try:
            with Image.open(path) as pil_image:
                image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        except Exception:
            return None
    elif image is None:
        return None
    return _resize_for_feature_extraction(image)


def _resize_for_feature_extraction(image: np.ndarray, max_dim: int = 600) -> np.ndarray:
    height, width = image.shape[:2]
    largest_dim = max(height, width)
    if largest_dim <= max_dim:
        return image
    scale = max_dim / largest_dim
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def _extract_features(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


@st.cache_resource(show_spinner=False)
def load_signature_templates() -> List[SignatureTemplate]:
    """Load all signature templates with features."""
    templates: List[SignatureTemplate] = []
    if not SIGNATURE_DIR.exists():
        return templates
    for image_path in sorted(SIGNATURE_DIR.glob("*")):
        if not image_path.is_file():
            continue
        raw_name = image_path.stem
        if "-" in raw_name:
            _, name_part = raw_name.split("-", 1)
        else:
            name_part = raw_name
        normalized = normalize_name(name_part.strip())
        gray_image = _load_image_as_gray(image_path)
        if gray_image is None:
            continue
        keypoints, descriptors = _extract_features(gray_image)
        templates.append(SignatureTemplate(name_key=normalized,
                                           display_name=name_part,
                                           path=image_path,
                                           image=gray_image,
                                           keypoints=keypoints,
                                           descriptors=descriptors))
    return templates


@st.cache_resource(show_spinner=False)
def load_photo_lookup() -> Dict[str, Path]:
    """Map normalized name keys to photo paths."""
    lookup: Dict[str, Path] = {}
    if not PHOTO_DIR.exists():
        return lookup
    for photo_path in sorted(PHOTO_DIR.glob("*")):
        if not photo_path.is_file():
            continue
        raw_name = photo_path.stem
        if "-" in raw_name:
            _, name_part = raw_name.split("-", 1)
        else:
            name_part = raw_name
        name_key = normalize_name(name_part.strip())
        lookup[name_key] = photo_path
    return lookup


def compute_match_score(
    uploaded_gray: np.ndarray,
    uploaded_kp: List[cv2.KeyPoint],
    uploaded_des: Optional[np.ndarray],
    template: SignatureTemplate,
) -> float:
    if uploaded_des is None or template.descriptors is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(uploaded_des, template.descriptors, k=2)
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    denominator = min(len(uploaded_kp), len(template.keypoints)) if template.keypoints else 0
    return float(len(good_matches) / denominator) if denominator else 0.0


def preprocess_upload(upload: io.BytesIO) -> Tuple[np.ndarray, List[cv2.KeyPoint], Optional[np.ndarray]]:
    """Read uploaded file, convert to grayscale, and extract features."""
    file_bytes = np.frombuffer(upload.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")
    image = _resize_for_feature_extraction(image)
    keypoints, descriptors = _extract_features(image)
    return image, keypoints, descriptors


def display_user_details(row: pd.Series, photo_lookup: Dict[str, Path], template_image: np.ndarray) -> None:
    """Display full user details with photo and signature."""
    st.subheader(row["Name"])
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Contact Details**")
        st.write(f"Email: {row['Email Address'] or 'N/A'}")
        st.write(f"Phone: {row['Contact number'] or 'N/A'}")
        st.markdown("**Address**")
        st.write(row["Address"] or "N/A")
        st.markdown("**Matched Signature Path**")
        st.write(str(row.get("MatchedSignaturePath", "N/A")))
    with col2:
        photo_path = photo_lookup.get(row["NameKey"])
        if photo_path and photo_path.exists():
            try:
                with Image.open(photo_path) as img:
                    st.image(img, caption="Identity Photo", use_column_width=True)
            except Exception:
                st.warning(f"Cannot open photo: {photo_path}")
        else:
            st.info("No identity photo found.")

    st.markdown("**Reference Signature**")
    st.image(template_image, caption="Stored Signature", use_column_width=False, clamp=True)


# --- MAIN APP ---
def main() -> None:
    st.set_page_config(page_title="Signature Verification", page_icon="✍️", layout="wide")
    st.title("Signature Verification Dashboard")
    st.write("Upload a signature to compare it against stored signatures.")

    dataset = load_dataset()
    templates = load_signature_templates()
    photo_lookup = load_photo_lookup()

    if not templates:
        st.error(f"No signature templates found in `{SIGNATURE_DIR}`.")
        return

    uploaded_file = st.file_uploader(
        "Upload a signature image (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is None:
        st.info("Please upload a signature image to begin.")
        return

    try:
        uploaded_image, uploaded_kp, uploaded_des = preprocess_upload(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.image(uploaded_image, caption="Uploaded Signature", use_column_width=False, clamp=True)

    # --- Compute match scores ---
    scores: List[Tuple[float, SignatureTemplate]] = []
    for template in templates:
        score = compute_match_score(uploaded_image, uploaded_kp, uploaded_des, template)
        scores.append((score, template))

    if not scores:
        st.warning("No matches found. Try a different image.")
        return

    scores.sort(key=lambda item: item[0], reverse=True)

    MATCH_THRESHOLD = st.slider(
        "Match acceptance threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.18,
        step=0.01,
        help="Minimum score required to consider a signature as matched."
    )

    st.markdown("### Top Matches")

    # Collect matches that meet threshold
    match_rows = []
    for score, template in scores:
        if score >= MATCH_THRESHOLD:
            # Flexible matching using normalized names
            matching_rows = dataset[dataset["NameKey"].str.contains(template.name_key)].copy()
            for idx, row in matching_rows.iterrows():
                row["MatchedSignaturePath"] = str(template.path)
                match_rows.append((row, template.image, score))

    if match_rows:
        for row, template_image, score in match_rows:
            st.markdown(f"**Match Score: {score:.3f}**")
            display_user_details(row, photo_lookup, template_image)
    else:
        # If no match exceeds threshold, still show the best candidate
        best_score, best_template = scores[0]
        matching_rows = dataset[dataset["NameKey"].str.contains(best_template.name_key)].copy()
        for idx, row in matching_rows.iterrows():
            row["MatchedSignaturePath"] = str(best_template.path)
            st.warning(f"No match exceeded threshold. Showing best candidate: {best_template.display_name} (score {best_score:.3f})")
            display_user_details(row, photo_lookup, best_template.image)


if __name__ == "__main__":
    main()
