# test_single_image.py
"""
Simple test script for single image processing.
"""

import cv2
import json
import sys
from detector import IntegratedDetector


def main():
    print("="*60)
    print("Integrated Multi-Person Detection System - Test Script")
    print("="*60)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test_image.jpg"
    
    print(f"\nLoading image: {image_path}")
    
    # Load image
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load image '{image_path}'")
        print("Please provide a valid image path.")
        return
    
    print(f"Image loaded: {frame.shape[1]}x{frame.shape[0]}")
    
    # Initialize detector
    print("\nInitializing detectors...")
    detector = IntegratedDetector()
    
    # Process frame
    print("\nProcessing image...")
    results = detector.process_frame(frame, frame_id=0)
    
    # Print results
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    
    print(f"\nScene: {results['scene']}")
    print(f"Number of persons detected: {len(results['detections'])}")
    
    for det in results['detections']:
        print(f"\n--- Person {det['id']} ---")
        print(f"  Position: {det['bbox_person']}")
        
        if det['bbox_face']:
            print(f"  Face detected: {det['bbox_face']}")
        else:
            print(f"  Face: Not detected")
        
        if det['attributes']:
            attr = det['attributes']
            print(f"  Age: {attr['age']}")
            print(f"  Gender: {attr['gender']}")
            print(f"  Emotion: {attr['emotion']}")
        else:
            print(f"  Attributes: Not available")
        
        if det['cloth'] and det['dress_color']:
            print(f"  Clothing: {det['dress_color']} {det['cloth']}")
        else:
            print(f"  Clothing: Not detected")
        
        print(f"  Pose: {det['pose']}")
    
    # Save JSON output
    json_output = "detection_results.json"
    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Full results saved to: {json_output}")
    
    # Visualize and save
    print("\nGenerating annotated image...")
    vis_frame = detector.visualize_results(frame, results)
    
    output_image = "output_annotated.jpg"
    cv2.imwrite(output_image, vis_frame)
    print(f"✓ Annotated image saved to: {output_image}")
    
    # Optionally display
    print("\nPress any key to close the display window...")
    cv2.imshow('Detection Results', vis_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)