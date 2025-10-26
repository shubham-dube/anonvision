from decision.decision_module import DecisionModule
from detection.face_detection import FaceDetector
import cv2

# Load image
image_path = "sample_images/group_photo1.jpg"
frame = cv2.imread(image_path)
frame = cv2.resize(frame, (640, 480))

# Detect faces
detector = FaceDetector(method="dnn")
faces = detector.detect_faces(frame)
print("Detected faces:", len(faces))

# Apply Decision Module
dm = DecisionModule(mode="crowd")  # choose your mode
faces_to_blur = dm.analyze(faces, frame)
print("Faces to blur:", len(faces_to_blur))

# Visualize
result = frame.copy()
for (x, y, w, h) in faces_to_blur:
    face_region = result[y:y+h, x:x+w]
    face_region = cv2.GaussianBlur(face_region, (51, 51), 30)
    result[y:y+h, x:x+w] = face_region

cv2.imshow("Blurred Faces", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
