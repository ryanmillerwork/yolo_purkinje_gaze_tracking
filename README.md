# Label Studio to YOLO Object Detection Converter

This script converts eye tracking annotations from Label Studio into YOLO object detection format, specifically designed for dual-Purkinje eye tracker training.

## What it does

The script automatically:
- Downloads annotated tasks from Label Studio via REST API
- Converts pupil, Purkinje-1, and Purkinje-4 mask annotations into separate YOLO object detection labels
- Handles missing elements gracefully (e.g., invisible Purkinje-4 reflections)
- Generates color-annotated preview images for quick quality control
- Organizes everything into a YOLO-compatible dataset structure

## Dual-Purkinje Eye Tracking Approach

This converter creates **three separate object classes** for maximum flexibility and precision:

- **Class 0**: `pupil_mask` - Large pupil region (may be partially occluded)
- **Class 1**: `purkinje1_mask` - First Purkinje reflection (usually visible)
- **Class 2**: `purkinje4_mask` - Fourth Purkinje reflection (often invisible)

This approach allows:
- **Fast inference** (~1-2ms) with standard YOLO object detection
- **Robust handling** of missing P4 reflections
- **Independent detection** of each component
- **Post-processing flexibility** for precise center estimation

## Features

- **Automated data pipeline**: No manual export/import needed
- **Handles missing elements**: Only exports completed annotations
- **Quality control previews**: Visual overlays show detected regions and bounding boxes
- **Standard YOLO compatibility**: Works with any YOLO object detection model
- **Secure configuration**: Credentials stored separately from code

## Prerequisites

- Python environment with Label Studio converter
- Label Studio instance running and accessible
- Project with brush label annotations for:
  - `pupil_mask`
  - `purkinje1_mask` 
  - `purkinje4_mask`

## Installation

1. Install required dependencies:
```bash
pip install label-studio-converter opencv-python numpy requests
```

2. Clone or download this script to your working directory

3. Create a `config.py` file with your Label Studio token:
```python
# Label Studio Configuration
LS_TOKEN = "your_label_studio_token_here"
```

## Configuration

Edit the settings in `ls2yolo.py`:

```python
LS_HOST    = "http://127.0.0.1:8080"               # Your Label Studio URL
PROJECT_ID = 7                                    # Your project ID number
OUT_ROOT   = pathlib.Path("yolo_dualp")            # Output dataset folder
IMG_ROOT   = "/label-studio/data/upload"           # LS upload directory path
```

### Finding your Label Studio token:
1. Open Label Studio in your browser
2. Go to Account & Settings
3. Copy the Access Token from the API section

### Finding your project ID:
- Check the URL when viewing your project: `/projects/{PROJECT_ID}/`
- Or check the project settings page

## Usage

Simply run the script:

```bash
source ~/labeling/bin/activate
python ls2yolo.py
```

The script will:
1. Download completed annotation tasks from your Label Studio project
2. Process each annotated image with proper RLE decoding
3. Create separate object detection labels for each mask
4. Generate preview images for quality control

## Output Structure

The script creates a complete YOLO dataset:

```
yolo_dualp/
├── images/
│   └── train/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   └── train/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── preview/
    ├── image1_preview.jpg
    ├── image2_preview.jpg
    └── ...
```

### Label Format

Each `.txt` file contains 1-3 lines in YOLO object detection format:
```
class_id center_x center_y width height
```

Example output:
```
2 0.534028 0.270000 0.023611 0.037778    # purkinje4: small reflection
1 0.476389 0.535556 0.041667 0.075556    # purkinje1: larger reflection  
0 0.533333 0.360000 0.308333 0.484444    # pupil: large region
```

Where:
- `class_id`: 0=pupil, 1=purkinje1, 2=purkinje4
- `center_x, center_y`: Bounding box center (normalized 0-1)
- `width, height`: Bounding box dimensions (normalized 0-1)

**Note**: Images may have 1-3 objects depending on visibility (e.g., missing P4 reflection)

### Preview Images

Color-coded overlays help verify the conversion:
- **Orange**: Pupil mask
- **Blue**: Purkinje-1 mask  
- **Purple**: Purkinje-4 mask
- **Green boxes**: Individual bounding boxes with class labels

## Troubleshooting

**"Cannot import decode_rle"**
- Install label-studio-converter: `pip install label-studio-converter`

**"FileNotFoundError: config.py"**
- Make sure you created the `config.py` file with your LS_TOKEN

**"HTTP 401 Unauthorized"**
- Verify your Label Studio token is correct and hasn't expired
- Check that the LS_HOST URL is accessible

**"No such file or directory" for images**
- Verify the IMG_ROOT path matches your Label Studio upload directory
- Check file permissions on the upload directory

## Security Note

The `config.py` file containing your Label Studio token is automatically excluded from git tracking via `.gitignore`. Never commit tokens or other credentials to version control.

## Integration with Training

The output dataset is ready to use with YOLOv8/YOLOv11 object detection:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolo11n.pt')  # Object detection, not pose

# Train the model
model.train(data='path/to/your/data.yaml', epochs=100)
```

Or with command line:
```bash
yolo detect train \
    model=yolo11n.pt \
    data=dualp_detection.yaml \
    imgsz=320 \
    epochs=20 \
    batch=16 \
    name=dual_purkinje_detector
```

### Post-Processing for Precision

After detection, implement precise center estimation:

1. **Pupil center**: Fit ellipse to detected region, handle occlusion
2. **Purkinje centers**: Use centroid of small detected regions
3. **Gaze calculation**: Apply dual-Purkinje mathematics

This two-stage approach (detection + fitting) provides both speed and precision for real-time eye tracking.