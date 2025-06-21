#!/usr/bin/env python3
"""
Test the trained dual-Purkinje model on video frames
"""

import cv2
import numpy as np
import pathlib
from ultralytics import YOLO

# Configuration
VIDEO_PATH = "eye_vids/OpenIris-2025Jun12-143843-Right.avi"  # Choose one video
MODEL_PATH = "runs/detect/dual_purkinje_640/weights/best.pt"
OUTPUT_DIR = pathlib.Path("video_inference_results")
FRAME_SKIP = 20  # Process every 20th frame
MAX_FRAMES = 50  # Limit total frames processed

# Class names for visualization
CLASS_NAMES = {0: "pupil", 1: "purkinje1", 2: "purkinje4"}
CLASS_COLORS = {
    0: (255, 128, 0),   # Orange for pupil
    1: (0, 128, 255),   # Blue for purkinje1  
    2: (60, 0, 255)     # Purple for purkinje4
}

def main():
    # Load the trained model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video info: {total_frames} frames at {fps:.1f} FPS")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    frame_count = 0
    processed_count = 0
    results_summary = []
    
    print(f"Processing every {FRAME_SKIP}th frame (max {MAX_FRAMES} frames)...")
    
    while cap.isOpened() and processed_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            print(f"Processing frame {frame_count}/{total_frames}")
            
            # Run inference
            results = model(frame, conf=0.25, imgsz=640, verbose=False)
            
            # Draw results on frame
            annotated_frame = frame.copy()
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box info
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        # Store detection info
                        detections.append({
                            'class': cls,
                            'class_name': CLASS_NAMES[cls],
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'center': [(x1+x2)//2, (y1+y2)//2]
                        })
                        
                        # Draw bounding box
                        color = CLASS_COLORS[cls]
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw center point
                        center = ((x1+x2)//2, (y1+y2)//2)
                        cv2.circle(annotated_frame, center, 3, color, -1)
                        
                        # Draw label
                        label = f"{CLASS_NAMES[cls]}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(annotated_frame, (x1, y1-label_size[1]-10), 
                                    (x1+label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Add frame info
            info_text = f"Frame {frame_count} | Detections: {len(detections)}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # Save annotated frame
            output_path = OUTPUT_DIR / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(output_path), annotated_frame)
            
            # Store results summary
            detection_summary = {
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'detections': detections,
                'counts': {name: sum(1 for d in detections if d['class'] == cls) 
                          for cls, name in CLASS_NAMES.items()}
            }
            results_summary.append(detection_summary)
            
            # Print detection summary
            counts = detection_summary['counts']
            print(f"  Found: {counts['pupil']} pupil, {counts['purkinje1']} P1, {counts['purkinje4']} P4")
            
            processed_count += 1
        
        frame_count += 1
    
    cap.release()
    
    # Print overall summary
    print(f"\n=== INFERENCE SUMMARY ===")
    print(f"Processed {processed_count} frames from {VIDEO_PATH}")
    
    # Count total detections by class
    total_counts = {name: 0 for name in CLASS_NAMES.values()}
    for summary in results_summary:
        for cls_name, count in summary['counts'].items():
            total_counts[cls_name] += count
    
    print(f"Total detections:")
    for cls_name, count in total_counts.items():
        print(f"  {cls_name}: {count}/{processed_count} frames ({count/processed_count*100:.1f}%)")
    
    # Detection consistency
    frames_with_pupil = sum(1 for s in results_summary if s['counts']['pupil'] > 0)
    frames_with_p1 = sum(1 for s in results_summary if s['counts']['purkinje1'] > 0) 
    frames_with_p4 = sum(1 for s in results_summary if s['counts']['purkinje4'] > 0)
    
    print(f"\nDetection rates:")
    print(f"  Pupil: {frames_with_pupil}/{processed_count} frames ({frames_with_pupil/processed_count*100:.1f}%)")
    print(f"  Purkinje1: {frames_with_p1}/{processed_count} frames ({frames_with_p1/processed_count*100:.1f}%)")
    print(f"  Purkinje4: {frames_with_p4}/{processed_count} frames ({frames_with_p4/processed_count*100:.1f}%)")
    
    print(f"\nAnnotated frames saved to: {OUTPUT_DIR}")
    print("Review the images to assess real-world performance!")

if __name__ == "__main__":
    main() 