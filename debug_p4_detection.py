#!/usr/bin/env python3
"""
Debug P4 detection - examine raw model outputs
"""

import cv2
import torch
from ultralytics import YOLO

# Configuration
VIDEO_PATH = "eye_vids/OpenIris-2025Jun12-143843-Right.avi"
MODEL_PATH = "runs/detect/dual_purkinje_640/weights/best.pt"

def main():
    # Load model
    model = YOLO(MODEL_PATH)
    
    # Get a few test frames where P4 should be visible
    cap = cv2.VideoCapture(VIDEO_PATH)
    test_frames = []
    frame_count = 0
    
    # Get frames 20, 40, 60 for testing
    target_frames = [20, 40, 60]
    
    while cap.isOpened() and len(test_frames) < len(target_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count in target_frames:
            test_frames.append((frame_count, frame))
            print(f"Captured frame {frame_count}")
        
        frame_count += 1
    
    cap.release()
    
    # Analyze each frame in detail
    for frame_num, frame in test_frames:
        print(f"\n=== ANALYZING FRAME {frame_num} ===")
        
        # Run inference with very low confidence to see ALL predictions
        results = model(frame, conf=0.01, imgsz=640, verbose=False)
        
        # Get raw predictions (before NMS)
        for i, result in enumerate(results):
            boxes = result.boxes
            if boxes is not None:
                print(f"Found {len(boxes)} detections:")
                
                # Sort by confidence and show all detections
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                
                sorted_indices = confidences.argsort()[::-1]  # High to low confidence
                
                class_names = {0: "pupil", 1: "purkinje1", 2: "purkinje4"}
                
                for idx in sorted_indices:
                    cls = int(classes[idx])
                    conf = confidences[idx]
                    x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
                    
                    print(f"  {class_names[cls]}: conf={conf:.4f}, bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            else:
                print("No detections found!")
        
        # Save frame with ALL detections (conf=0.01) for visual inspection
        annotated = frame.copy()
        
        # Draw all detections with confidence > 0.01
        results_low_conf = model(frame, conf=0.01, imgsz=640, verbose=False)
        
        colors = {0: (255, 128, 0), 1: (0, 128, 255), 2: (60, 0, 255)}
        class_names = {0: "pupil", 1: "purkinje1", 2: "purkinje4"}
        
        for result in results_low_conf:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    color = colors[cls]
                    
                    # Draw box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with confidence
                    label = f"{class_names[cls]}: {conf:.3f}"
                    cv2.putText(annotated, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save debug frame
        output_path = f"debug_p4_frame_{frame_num}.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"Saved debug frame: {output_path}")
        
        # Special focus on P4 class
        p4_detections = []
        for result in results_low_conf:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 2:  # P4 class
                        conf = float(box.conf[0])
                        p4_detections.append(conf)
        
        if p4_detections:
            print(f"P4 detections found: {len(p4_detections)}")
            print(f"P4 confidences: {p4_detections}")
            print(f"Highest P4 confidence: {max(p4_detections):.4f}")
        else:
            print("NO P4 detections found at any confidence level!")
    
    print(f"\nNext steps:")
    print(f"1. Check debug_p4_frame_*.jpg files")
    print(f"2. Look for visible P4 reflections that aren't detected")
    print(f"3. Note the confidence levels of any P4 detections")

if __name__ == "__main__":
    main() 