# test_blurring.py
import cv2
from detector import IntegratedDetector
from decision.decision_module import DecisionModule
from blurring.blurring_module import FaceBlurrer, SelectiveBlurPipeline

def test_blurring_webcam():
    print("Initializing pipeline...")
    
    # Initialize components
    detector = IntegratedDetector()
    decision = DecisionModule(mode='all')  # Blur all faces
    blurrer = FaceBlurrer(blur_type='gaussian', blur_intensity='medium')
    pipeline = SelectiveBlurPipeline(detector, decision, blurrer)
    
    print("✅ Pipeline ready!")
    print("\nControls:")
    print("  Q - Quit")
    print("  G - Gaussian blur")
    print("  P - Pixelate")
    print("  M - Mosaic")
    print("  B - Black box")
    print("  1/2/3 - Low/Medium/High intensity")
    
    # Process webcam
    pipeline.process_webcam(show_debug=True)
    
    print("✅ Module 3 test complete!")

if __name__ == "__main__":
    test_blurring_webcam()