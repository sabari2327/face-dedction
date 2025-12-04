"""
Interactive face matching: Upload an image, then match faces in live camera feed.
"""

import os
import cv2
import pickle
import argparse
import numpy as np
from imutils import paths
from pathlib import Path


def load_models():
    """Load the trained face detector, embedder, and recognizer models."""
    prototxt_path = os.path.join('face_detection_model', 'deploy.prototxt')
    weights_path = os.path.join('face_detection_model', 'res10_300x300_ssd_iter_140000.caffemodel')
    embeddings_path = os.path.join('output', 'openface_nn4.small2.v1.t7')
    
    if not os.path.exists(prototxt_path):
        print(f"ERROR: Face detector prototxt not found at {prototxt_path}")
        return None, None, None, None
    if not os.path.exists(weights_path):
        print(f"ERROR: Face detector weights not found at {weights_path}")
        return None, None, None, None
    if not os.path.exists(embeddings_path):
        print(f"ERROR: Embeddings model not found at {embeddings_path}")
        return None, None, None, None
    
    # Load face detector
    net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
    
    # Load face embedder
    embedder = cv2.dnn.readNetFromTorch(embeddings_path)
    
    # Load recognizer and label encoder
    recognizer_path = os.path.join('output', 'recognizer')
    le_path = os.path.join('output', 'le.pickle')
    
    if not os.path.exists(recognizer_path):
        print(f"ERROR: Trained recognizer not found at {recognizer_path}")
        print("Please run: python train_model.py")
        return None, None, None, None
    if not os.path.exists(le_path):
        print(f"ERROR: Label encoder not found at {le_path}")
        return None, None, None, None
    
    recognizer = pickle.load(open(recognizer_path, 'rb'))
    le = pickle.load(open(le_path, 'rb'))
    
    return net, embedder, recognizer, le


def get_face_embedding(frame, net, embedder, confidence=0.5):
    """
    Detect faces in frame and return their bounding boxes and embeddings.
    
    Returns:
        List of tuples: (startX, startY, endX, endY, embedding)
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104.0, 177.0, 123.0], False, False)
    
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    
    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        
        if conf < confidence:
            continue
        
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)
        
        face = frame[startY:endY, startX:endX]
        
        if face.shape[0] < 20 or face.shape[1] < 20:
            continue
        
        face = cv2.resize(face, (96, 96))
        face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), [0, 0, 0], True, False)
        
        embedder.setInput(face_blob)
        vec = embedder.forward()
        
        faces.append((startX, startY, endX, endY, vec))
    
    return faces


def upload_image_dialog():
    """Ask user for image path to upload."""
    print("\n" + "="*60)
    print("FACE MATCHING - Step 1: Upload Image")
    print("="*60)
    print("\nPlease provide the path to an image containing the face you want to match.")
    print("Examples: images\\openface.jpg, C:\\path\\to\\face.jpg")
    print("\nOr press Enter to use default: images\\openface.jpg")
    
    image_path = input("\nImage path: ").strip()
    
    if not image_path:
        image_path = os.path.join('images', 'openface.jpg')
    
    if not os.path.exists(image_path):
        print(f"\nERROR: Image not found at '{image_path}'")
        return None
    
    return image_path


def get_name_dialog():
    """Ask user for the name of the person in the uploaded image."""
    print("\n" + "="*60)
    print("Step 2: Enter Name")
    print("="*60)
    print("\nEnter the name of the person in the uploaded image:")
    
    name = input("Name: ").strip()
    
    if not name:
        print("ERROR: Name cannot be empty")
        return None
    
    return name


def get_face_from_image(image_path, net, embedder, confidence=0.5):
    """Extract face embedding from uploaded image."""
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        return None, None
    
    print("Detecting faces in uploaded image...")
    faces = get_face_embedding(image, net, embedder, confidence)
    
    if not faces:
        print("ERROR: No faces detected in the uploaded image")
        return None, None
    
    if len(faces) > 1:
        print(f"\nWARNING: {len(faces)} faces found. Using the first one.")
    
    startX, startY, endX, endY, embedding = faces[0]
    
    print(f"Face detected: ({startX}, {startY}) to ({endX}, {endY})")
    print(f"Face embedding extracted successfully")
    
    # Draw bounding box on image and show
    output_image = image.copy()
    cv2.rectangle(output_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(output_image, "Uploaded Face", (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return embedding, output_image


def euclidean_distance(vec_a, vec_b):
    """Calculate Euclidean distance between two vectors."""
    return np.linalg.norm(vec_a - vec_b)


def run_live_matching(target_embedding, target_name, net, embedder, recognizer, le, 
                      confidence=0.5, threshold=0.6, camera_src=0):
    """
    Run live camera feed and match faces against target embedding.
    
    Args:
        target_embedding: The embedding of the face to match
        target_name: Name of the person to match
        net: Face detection network
        embedder: Face embedding network
        recognizer: Trained SVM recognizer (for optional classification)
        le: Label encoder (for optional classification)
        confidence: Detection confidence threshold (0-1)
        threshold: Distance threshold for matching (lower = stricter)
        camera_src: Camera source (0 for default webcam)
    """
    print("\n" + "="*60)
    print("Step 3: Live Face Matching")
    print("="*60)
    print(f"\nMatching against: {target_name}")
    print(f"Distance threshold: {threshold} (lower = stricter matching)")
    print("\nStarting camera...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save snapshot")
    print("-" * 60)
    
    vs = cv2.VideoCapture(camera_src)
    
    if not vs.isOpened():
        print(f"ERROR: Could not open camera {camera_src}")
        return
    
    writer = None
    fps = 30
    snapshot_count = 0
    
    try:
        while True:
            ret, frame = vs.read()
            
            if not ret:
                print("ERROR: Failed to grab frame from camera")
                break
            
            frame = cv2.resize(frame, (640, 480))
            h, w = frame.shape[:2]
            
            # Detect faces
            faces = get_face_embedding(frame, net, embedder, confidence)
            
            match_count = 0
            
            for startX, startY, endX, endY, embedding in faces:
                # Calculate distance to target embedding
                distance = euclidean_distance(target_embedding, embedding)
                
                # Determine if it's a match
                is_match = distance < threshold
                
                # Draw bounding box
                color = (0, 255, 0) if is_match else (0, 0, 255)  # Green for match, Red for no match
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                # Draw label
                if is_match:
                    label = f"{target_name} (Match!)"
                    match_count += 1
                else:
                    label = f"No Match (dist: {distance:.2f})"
                
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show match count
            status_text = f"Matches: {match_count} | Faces: {len(faces)}"
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, target_name, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow("Face Matching - Press 'q' to quit, 's' to save", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuit requested by user")
                break
            elif key == ord('s'):
                snapshot_path = os.path.join('output', f'match_snapshot_{snapshot_count}.jpg')
                cv2.imwrite(snapshot_path, frame)
                print(f"Snapshot saved: {snapshot_path}")
                snapshot_count += 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        vs.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("\nCamera closed")


def main():
    parser = argparse.ArgumentParser(description='Upload image and match faces in live camera feed')
    parser.add_argument('--image', type=str, help='Path to image to upload (if not provided, will ask)')
    parser.add_argument('--name', type=str, help='Name of person in image (if not provided, will ask)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence (0-1)')
    parser.add_argument('--threshold', type=float, default=0.6, help='Matching distance threshold (lower = stricter)')
    parser.add_argument('--src', type=str, default='0', help='Camera source (0, 1, 2, ... or video file path)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("INTERACTIVE FACE MATCHER")
    print("="*60)
    
    # Load models
    print("\nLoading models...")
    net, embedder, recognizer, le = load_models()
    if net is None:
        print("\nERROR: Failed to load models. Exiting.")
        return
    print("Models loaded successfully")
    
    # Get image path
    if args.image:
        image_path = args.image
    else:
        image_path = upload_image_dialog()
        if image_path is None:
            return
    
    # Get face embedding from uploaded image
    target_embedding, output_image = get_face_from_image(image_path, net, embedder, args.confidence)
    if target_embedding is None:
        return
    
    # Show uploaded image with detected face
    if output_image is not None:
        cv2.imshow("Uploaded Image - Face Detected", output_image)
        print("\nShowing uploaded image. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow("Uploaded Image - Face Detected")
    
    # Get name
    if args.name:
        target_name = args.name
    else:
        target_name = get_name_dialog()
        if target_name is None:
            return
    
    print(f"\n✓ Ready to match: {target_name}")
    
    # Parse camera source
    try:
        camera_src = int(args.src)
    except ValueError:
        camera_src = args.src  # Assume it's a video file path
    
    # Run live matching
    input("\nPress Enter to start camera...")
    run_live_matching(target_embedding, target_name, net, embedder, recognizer, le,
                      args.confidence, args.threshold, camera_src)
    
    print("\n" + "="*60)
    print("Face matching session ended")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
