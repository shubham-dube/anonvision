# detection/attribute_extractor.py
from deepface import DeepFace
import cv2

class AttributeExtractor:
    """Extract facial attributes using DeepFace."""
    
    def __init__(self):
        """Initialize attribute extractor."""
        pass
    
    def analyze(self, face_crop):
        """
        Analyze facial attributes.
        
        Args:
            face_crop: Cropped face image (BGR)
            
        Returns:
            Dictionary with age, gender, emotion (and optionally race)
        """
        try:
            result = DeepFace.analyze(
                face_crop,
                actions=['age', 'gender', 'emotion', 'race'], 
                enforce_detection=False,
                silent=True
            )

            # DeepFace returns a list with one result per detected face
            info = result[0]

            return {
                "age": int(info.get('age', 0)),
                "gender": info.get('dominant_gender', 'unknown'),
                "dominant_emotion": info.get('dominant_emotion', 'neutral'),
                "emotions": info.get('emotion', {}),  # full emotion probability dict
                # Uncomment below if you add 'race' in actions
                "dominant_race": info.get('dominant_race', 'unknown'),
                "races": info.get('race', {})
            }
        except Exception as e:
            print(f"⚠️ Attribute extraction error: {e}")
            return {
                "age": None,
                "gender": None,
                "dominant_emotion": None,
                "emotions": {}
            }


if __name__ == "__main__":
    # Path to the test image in the project root
    image_path = os.path.join(os.getcwd(), "test_img.jpg")  # adjust if it's .png or other extension

    if not os.path.exists(image_path):
        print(f"Test image not found at {image_path}")
    else:
        img = cv2.imread(image_path)
        if img is None:
            print("Failed to read image. Check file format.")
        else:
            extractor = AttributeExtractor()
            result = extractor.analyze(img)
            print("Analysis Result:")
            print(result)
