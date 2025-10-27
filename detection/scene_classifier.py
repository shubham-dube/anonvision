# detection/scene_classifier.py
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os

class SceneClassifier:
    """Classify scene using the pretrained Places365 model."""
    
    def __init__(self):
        """Initialize scene classifier."""
        import torchvision.models as models
        from torch.hub import load_state_dict_from_url

        # --- Load pretrained Places365 model ---
        model_file = 'resnet50_places365.pth.tar'
        model_url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
        model_dir = 'detection/models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_file)

        if not os.path.exists(model_path):
            print("🔽 Downloading Places365 ResNet50 model...")
            torch.hub.download_url_to_file(model_url, model_path)
            print("✅ Downloaded model:", model_path)

        # Load model architecture
        self.model = models.resnet50(num_classes=365)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # --- Load class labels ---
        label_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        labels_path = 'detection/categories_places365.txt'
        if not os.path.exists(labels_path):
            print("🔽 Downloading Places365 category labels...")
            import urllib.request
            urllib.request.urlretrieve(label_url, labels_path)
        self.classes = [line.strip().split(' ')[0][3:] for line in open(labels_path)]

        # --- Image preprocessing ---
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict_scene(self, frame):
        """
        Predict scene category for a given frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            Scene label string
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Transform and predict
        input_tensor = self.transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(input_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()

        if pred_idx >= len(self.classes):
            return "Unknown Scene"

        scene_label = self.classes[pred_idx]
        return scene_label
