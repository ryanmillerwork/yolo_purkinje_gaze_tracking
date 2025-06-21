# Label Studio to YOLO-Pose Converter

This script converts eye tracking annotations from Label Studio into YOLO-Pose format, specifically designed for pupil and Purkinje reflection detection.

## What it does

The script automatically:
- Downloads the latest JSON export from Label Studio via REST API
- Converts pupil, Purkinje-1, and Purkinje-4 mask annotations into YOLO-Pose keypoint labels
- Generates color-annotated preview images for quick quality control
- Organizes everything into a YOLO-compatible dataset structure

## Features

- **Automated data pipeline**: No manual export/import needed
- **Quality control previews**: Visual overlays show detected regions and keypoints
- **YOLO-Pose compatibility**: Ready-to-use format for training pose estimation models
- **Secure configuration**: Credentials stored separately from code

## Prerequisites

- Python environment with access to your MobileSAM backend
- Label Studio instance running and accessible
- Project with brush label annotations for:
  - `pupil_mask`
  - `purkinje1_mask` 
  - `purkinje4_mask`

## Installation

1. Install required dependencies:
```bash
pip install label-studio-sdk opencv-python numpy requests
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
1. Download all tasks from your Label Studio project
2. Process each annotated image
3. Create the YOLO dataset structure
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

Each `.txt` file contains YOLO-Pose format labels:
```
class_id center_x center_y width height kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis kp3_x kp3_y kp3_vis
```

Where:
- `class_id`: Always 0 (single class for eye region)
- `center_x, center_y, width, height`: Bounding box (normalized 0-1)
- `kp1, kp2, kp3`: Keypoints for pupil, Purkinje-1, Purkinje-4 (normalized 0-1)
- `kp_vis`: Visibility flag (always 2 = visible)

### Preview Images

Color-coded overlays help verify the conversion:
- **Orange**: Pupil mask
- **Blue**: Purkinje-1 mask  
- **Purple**: Purkinje-4 mask
- **Green**: Bounding box
- **Black crosses**: Keypoint centers

## Troubleshooting

**"Export zip had no JSON file inside"**
- Check that your Label Studio project has completed annotations
- Verify the project ID is correct

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

The output dataset is ready to use with YOLOv8 pose estimation:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')

# Train the model
model.train(data='path/to/your/data.yaml', epochs=100)
```

You'll need to create a `data.yaml` file pointing to your dataset paths. 