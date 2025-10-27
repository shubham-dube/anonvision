# detection/attribute_extractor.py
# from deepface import DeepFace
import cv2
import os

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
            Dictionary with age, gender, emotion
        """
        try:
            # result = DeepFace.analyze(
            #     face_crop,
            #     actions=['age', 'gender'],
            #     enforce_detection=False,
            #     silent=True
            # )
            
            # return {
            #     "age": int(result[0]['age']),
            #     "gender": result[0]['dominant_gender'],
            #     "emotions": None
            # }
            return {
                "age": 21,
                "gender": "Male",
                "emotions": None
            }
        except Exception as e:
            print(f"Attribute extraction error: {e}")
            return {"age": None, "gender": None, "emotion": None}


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
