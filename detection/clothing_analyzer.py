# detection/clothing_analyzer.py
# Advanced clothing color analyzer (no type classification).
# Returns detailed color name from clothing region.
# Requires: pip install opencv-python-headless scikit-learn numpy webcolors

import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors

# -------------------------------------------------------------------
# Build color name lookup dynamically (compatible with all versions)
# -------------------------------------------------------------------

def _get_color_name_map():
    """Return hex→name dictionary safely across webcolors versions."""
    if hasattr(webcolors, "CSS3_NAMES_TO_HEX"):
        names_to_hex = webcolors.CSS3_NAMES_TO_HEX
        return {v: k for k, v in names_to_hex.items()}
    elif hasattr(webcolors, "HTML4_NAMES_TO_HEX"):
        names_to_hex = webcolors.HTML4_NAMES_TO_HEX
        return {v: k for k, v in names_to_hex.items()}
    else:
        # fallback minimal palette
        return {
            "#000000": "black",
            "#ffffff": "white",
            "#ff0000": "red",
            "#00ff00": "lime",
            "#0000ff": "blue",
            "#ffff00": "yellow",
            "#ffa500": "orange",
            "#800080": "purple",
            "#808080": "gray",
            "#00ffff": "cyan",
            "#ffc0cb": "pink"
        }

HEX_TO_NAMES = _get_color_name_map()


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def closest_color_name(requested_rgb):
    """Find the closest named color from CSS3/HTML color dictionary."""
    min_dist, closest_name = float("inf"), None
    for hex_code, name in HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
        dist = (r_c - requested_rgb[0]) ** 2 + (g_c - requested_rgb[1]) ** 2 + (b_c - requested_rgb[2]) ** 2
        if dist < min_dist:
            min_dist, closest_name = dist, name
    return closest_name or "unknown"


def refine_color_name(name):
    """Make color names more readable and consistent."""
    name = name.lower().replace("-", " ")
    name = name.replace("grey", "gray")
    return name.strip()


def get_dominant_colors(image_bgr, top_k=3):
    """
    Extract top K dominant colors from a clothing crop.
    Returns a list of RGB tuples.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)

    # Remove shadows and highlights
    brightness = np.mean(pixels, axis=1)
    valid = (brightness > 25) & (brightness < 235)
    pixels = pixels[valid]

    if len(pixels) < 30:
        pixels = image_rgb.reshape(-1, 3)

    # Cluster colors
    k = min(top_k, max(1, len(pixels) // 500))
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    order = np.argsort(-counts)
    return [tuple(centers[i]) for i in order[:top_k]]


# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------

def analyze_clothing(frame, person_bbox):
    """
    Analyze clothing color.
    Returns:
        {
          "color": "light blue with gray tone",
          "clothing_type": None
        }
    """
    x1, y1, x2, y2 = map(int, person_bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return {"color": "unknown", "clothing_type": None}

    h, w = crop.shape[:2]
    torso_crop = crop[int(h * 0.15):int(h * 0.65), int(w * 0.15):int(w * 0.85)]
    if torso_crop.size == 0:
        torso_crop = crop

    try:
        top_colors = get_dominant_colors(torso_crop, top_k=3)
        color_names = [refine_color_name(closest_color_name(rgb)) for rgb in top_colors]

        # Create descriptive name (main + secondary tones)
        if len(color_names) == 1:
            final_color = color_names[0]
        elif len(color_names) >= 2:
            if color_names[1].split()[0] not in color_names[0]:
                final_color = f"{color_names[0]} with {color_names[1]} tone"
            else:
                final_color = color_names[0]
        else:
            final_color = "unknown"

        return {"color": final_color, "clothing_type": None}

    except Exception as e:
        print(f"[ClothingAnalyzer] error: {e}")
        return {"color": "unknown", "clothing_type": None}
