"""
Interactive face matching with GUI file picker.
Upload image via file dialog, then match faces in live camera feed with matching percentage.
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox


def load_models():
    """Load the face detector and embedder models."""
    prototxt_path = os.path.join('face_detection_model', 'deploy.prototxt')
    weights_path = os.path.join('face_detection_model', 'res10_300x300_ssd_iter_140000.caffemodel')
    embeddings_path = 'openface_nn4.small2.v1.t7'
    
    if not os.path.exists(prototxt_path):
        print(f"ERROR: Face detector prototxt not found at {prototxt_path}")
        return None, None
    if not os.path.exists(weights_path):
        print(f"ERROR: Face detector weights not found at {weights_path}")
        return None, None
    if not os.path.exists(embeddings_path):
        print(f"ERROR: Embeddings model not found at {embeddings_path}")
        return None, None
    
    # Load face detector
    net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
    
    # Load face embedder
    embedder = cv2.dnn.readNetFromTorch(embeddings_path)
    
    return net, embedder


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


def pick_image_dialog():
    """Open file picker dialog to select image."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select an image with a face",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path if file_path else None


def get_name_dialog():
    """Ask user for the name of the person in the uploaded image."""
    root = tk.Tk()
    root.withdraw()
    
    name = simpledialog.askstring(
        "Enter Name",
        "Enter the name of the person in the image:",
        parent=root
    )
    
    root.destroy()
    return name if name else None


def get_face_from_image(image_path, net, embedder, confidence=0.5):
    """Extract face embedding from uploaded image."""
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        messagebox.showerror("Error", f"Could not load image from:\n{image_path}")
        return None, None
    
    print("Detecting faces in uploaded image...")
    faces = get_face_embedding(image, net, embedder, confidence)
    
    if not faces:
        print("ERROR: No faces detected in the uploaded image")
        messagebox.showerror("Error", "No faces detected in the uploaded image.\nPlease choose another image.")
        return None, None
    
    if len(faces) > 1:
        print(f"\nWARNING: {len(faces)} faces found. Using the first one.")
    
    startX, startY, endX, endY, embedding = faces[0]
    
    print(f"Face detected: ({startX}, {startY}) to ({endX}, {endY})")
    print(f"Face embedding extracted successfully")
    
    # Draw bounding box on image
    output_image = image.copy()
    cv2.rectangle(output_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(output_image, "Uploaded Face", (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return embedding, output_image


def euclidean_distance(vec_a, vec_b):
    """Calculate Euclidean distance between two vectors."""
    return np.linalg.norm(vec_a - vec_b)


def run_live_matching(target_embedding, target_name, net, embedder, 
                      confidence=0.5, threshold=0.6, camera_src=0):
    """
    Run live camera feed and match faces against target embedding.
    
    Args:
        target_embedding: The embedding of the face to match
        target_name: Name of the person to match
        net: Face detection network
        embedder: Face embedding network
        confidence: Detection confidence threshold (0-1)
        threshold: Distance threshold for matching (lower = stricter)
        camera_src: Camera source (0 for default webcam)
    """
    print("\n" + "="*60)
    print("Step 3: Live Face Matching")
    print("="*60)
    print(f"\nMatching against: {target_name}")
    print(f"Distance threshold: {threshold} (lower = stricter matching)")
    print("Matching percentage: Shows how similar detected faces are")
    print("\nStarting camera...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save snapshot")
    print("-" * 60)
    
    vs = cv2.VideoCapture(camera_src)
    
    if not vs.isOpened():
        print(f"ERROR: Could not open camera {camera_src}")
        messagebox.showerror("Camera Error", f"Could not open camera {camera_src}")
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
                
                # Calculate matching percentage (inverse of normalized distance)
                # Distance ranges from ~0.3 (very similar) to ~1.5 (very different)
                match_percentage = max(0, min(100, (1.0 - distance) * 100))
                
                # Determine if it's a match
                is_match = distance < threshold
                
                # Draw bounding box
                color = (0, 255, 0) if is_match else (0, 0, 255)  # Green for match, Red for no match
                thickness = 3 if is_match else 2
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, thickness)
                
                # Draw label with matching percentage
                if is_match:
                    label = f"{target_name} - {match_percentage:.1f}% MATCH"
                    match_count += 1
                else:
                    label = f"{match_percentage:.1f}% Match"
                
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
    parser = argparse.ArgumentParser(description='Upload image via file picker and match faces in live camera')
    parser.add_argument('--image', type=str, help='Path to image (default: opens file picker)')
    parser.add_argument('--name', type=str, help='Name of person (default: asks via dialog)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence (0-1)')
    parser.add_argument('--threshold', type=float, default=0.6, help='Matching distance threshold (lower = stricter)')
    parser.add_argument('--src', type=str, default='0', help='Camera source (0, 1, 2, ... or video file path)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("INTERACTIVE FACE MATCHER (GUI)")
    print("="*60)
    print("\nNo pre-training needed!")
    print("Upload any image from your computer and match faces in real-time.")
    
    # Load models
    print("\nLoading models...")
    net, embedder = load_models()
    if net is None:
        print("\nERROR: Failed to load models. Exiting.")
        messagebox.showerror("Error", "Failed to load AI models.\nMake sure all model files are present.")
        return
    print("Models loaded successfully")
    
    # Get image path
    if args.image:
        image_path = args.image
        print(f"\nUsing image: {image_path}")
    else:
        print("\nOpening file picker...")
        image_path = pick_image_dialog()
        if not image_path:
            print("No image selected. Exiting.")
            messagebox.showwarning("Cancelled", "No image was selected.")
            return
    
    # Get face embedding from uploaded image
    target_embedding, output_image = get_face_from_image(image_path, net, embedder, args.confidence)
    if target_embedding is None:
        return
    
    # Show uploaded image with detected face
    if output_image is not None:
        try:
            cv2.imshow("Uploaded Image - Face Detected", output_image)
            print("\nShowing uploaded image. Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("\nFace detected in uploaded image. Proceeding to camera...")
    
    # Get name
    if args.name:
        target_name = args.name
    else:
        target_name = get_name_dialog()
        if not target_name:
            print("No name entered. Exiting.")
            messagebox.showwarning("Cancelled", "No name was entered.")
            return
    
    print(f"\n✓ Ready to match: {target_name}")
    
    # Parse camera source
    try:
        camera_src = int(args.src)
    except ValueError:
        camera_src = args.src  # Assume it's a video file path
    
    # Run live matching
    print("\nPress Enter to start camera...")
    input()
    run_live_matching(target_embedding, target_name, net, embedder,
                      args.confidence, args.threshold, camera_src)
    
    print("\n" + "="*60)
    print("Face matching session ended")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
