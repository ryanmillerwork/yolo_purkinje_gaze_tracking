#!/usr/bin/env python3
"""
Test different confidence thresholds to optimize Purkinje detection
"""

import cv2
import numpy as np
import pathlib
from ultralytics import YOLO

# Configuration
VIDEO_PATH = "eye_vids/OpenIris-2025Jun12-143843-Right.avi"
MODEL_PATH = "runs/detect/dual_purkinje_640/weights/best.pt"
FRAME_SKIP = 20
MAX_FRAMES = 20  # Just test on first 20 frames

# Test different confidence thresholds
CONFIDENCE_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

# Class names
CLASS_NAMES = {0: "pupil", 1: "purkinje1", 2: "purkinje4"}

def test_confidence_threshold(model, frames, conf_threshold):
    """Test a specific confidence threshold on the frames."""
    results = {name: 0 for name in CLASS_NAMES.values()}
    total_detections = 0
    
    for frame in frames:
        # Run inference
        predictions = model(frame, conf=conf_threshold, imgsz=640, verbose=False)
        
        for result in predictions:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = CLASS_NAMES[cls]
                    results[class_name] += 1
                    total_detections += 1
    
    return results, total_detections

def main():
    # Load model
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    
    # Load test frames
    print("Loading test frames...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    frame_count = 0
    
    while len(frames) < MAX_FRAMES and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % FRAME_SKIP == 0:
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    print(f"Loaded {len(frames)} test frames")
    
    # Test different confidence thresholds
    print(f"\n{'Conf':<6} {'Pupil':<6} {'P1':<6} {'P4':<6} {'Total':<6} {'P1%':<6} {'P4%':<6}")
    print("-" * 45)
    
    for conf in CONFIDENCE_THRESHOLDS:
        results, total = test_confidence_threshold(model, frames, conf)
        
        pupil_rate = results['pupil'] / len(frames) * 100
        p1_rate = results['purkinje1'] / len(frames) * 100  
        p4_rate = results['purkinje4'] / len(frames) * 100
        
        print(f"{conf:<6.2f} {results['pupil']:<6} {results['purkinje1']:<6} {results['purkinje4']:<6} {total:<6} {p1_rate:<6.1f} {p4_rate:<6.1f}")
    
    print(f"\nRecommendation: Look for the confidence threshold that maximizes P1 detection")
    print(f"without too many false positives. Consider conf=0.1 or conf=0.15 if they")
    print(f"significantly improve Purkinje detection rates.")

if __name__ == "__main__":
    main() 