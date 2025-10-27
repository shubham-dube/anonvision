# detection/clothing_analyzer.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
import torch
from PIL import Image
from torchvision import transforms
import clip

# Initialize CLIP model globally for efficiency
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

CLOTHING_LABELS = [
    "shirt", "t-shirt", "jacket", "hoodie", "coat", "sweater",
    "dress", "suit", "uniform", "kurta", "saree", "traditional wear"
]

COLOR_MAP = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "orange": (255, 165, 0),
    "brown": (139, 69, 19),
    "pink": (255, 192, 203),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
}


def rgb_to_color_name(rgb):
    """Convert RGB to nearest color name."""
    def color_distance(c1, c2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    
    min_dist, name = float("inf"), "unknown"
    for cname, cval in COLOR_MAP.items():
        dist = color_distance(rgb, cval)
        if dist < min_dist:
            min_dist, name = dist, cname
    
    return name


def get_dominant_color(image, k=3):
    """
    Extract dominant color from clothing region.
    
    Args:
        image: BGR image
        k: Number of clusters for K-means
        
    Returns:
        Color name string
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    
    # Filter out very dark (shadows) and very bright (highlights) pixels
    brightness = np.mean(pixels, axis=1)
    mask = (brightness > 20) & (brightness < 235)
    filtered_pixels = pixels[mask]
    
    if len(filtered_pixels) < 10:
        filtered_pixels = pixels
    
    kmeans = KMeans(n_clusters=min(k, len(filtered_pixels)), n_init=10, random_state=42)
    kmeans.fit(filtered_pixels)
    
    # Get the most common cluster
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = labels[np.argmax(counts)]
    dominant_rgb = kmeans.cluster_centers_[dominant_cluster].astype(int)
    
    return rgb_to_color_name(tuple(dominant_rgb))


def detect_clothing_type(image_bgr):
    """
    Classify clothing type using CLIP.
    
    Args:
        image_bgr: BGR image of clothing region
        
    Returns:
        Clothing type string
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    image_input = preprocess(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_inputs = clip.tokenize(CLOTHING_LABELS).to(device)
        text_features = clip_model.encode_text(text_inputs)
        
        similarities = (image_features @ text_features.T).squeeze(0)
        best_idx = similarities.argmax().item()
    
    return CLOTHING_LABELS[best_idx]


def analyze_clothing(frame, person_bbox):
    """
    Analyze clothing color and type.
    
    Args:
        frame: Full frame (BGR)
        person_bbox: [x1, y1, x2, y2] bounding box
        
    Returns:
        Dictionary with color and clothing_type
    """
    x1, y1, x2, y2 = map(int, person_bbox)
    
    # Ensure valid crop
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    person_crop = frame[y1:y2, x1:x2]
    
    if person_crop.size == 0:
        return {"color": "unknown", "clothing_type": "unknown"}
    
    h, w = person_crop.shape[:2]
    
    # Focus on torso region (upper-middle body)
    torso_crop = person_crop[int(h*0.15):int(h*0.65), int(w*0.15):int(w*0.85)]
    
    if torso_crop.size == 0:
        torso_crop = person_crop
    
    try:
        color = get_dominant_color(torso_crop)
        cloth_type = detect_clothing_type(torso_crop)
        
        return {"color": color, "clothing_type": cloth_type}
    except Exception as e:
        print(f"Clothing analysis error: {e}")
        return {"color": "unknown", "clothing_type": "unknown"}