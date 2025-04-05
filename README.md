# YOLOv5-VGG16 Screw Detection System

## Overview

This project presents an integrated solution for automated screw detection and condition assessment using deep learning. The system combines YOLOv5 for object detection and VGG16 for classification to identify screws and determine their condition, specifically detecting rusted ("shengxiu") or slipped ("huasi") screws.

## Key Features

### Two-Stage Detection Pipeline
- **Object Detection**: YOLOv5 locates and identifies screws in images with high precision
- **Condition Classification**: VGG16 analyzes detected screws to classify their condition (rusted/slipped)

### Real-time Processing
- Integrates with camera feed for live screw inspection
- User-friendly GUI interface for interactive use
- Multi-threaded processing for responsive performance

### Comprehensive Workflow
- Image capture from webcam
- Automated object detection
- Detailed condition assessment
- Results visualization and logging

## Technical Architecture

### Models
- **YOLOv5**: State-of-the-art object detection model customized for screw detection
- **VGG16**: Pre-trained CNN fine-tuned for screw condition classification

### Implementation Details
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Image Processing**: OpenCV
- **GUI**: Tkinter

## Usage

1. Run `main.py` to launch the application interface
2. Use the camera preview to position screws in frame
3. Click the "Take Photo" button to capture an image
4. The system will automatically:
   - Detect screws using YOLOv5 (Script 1)
   - Crop detected screws (Script 2)
   - Classify screw conditions using VGG16 (Script 3)
5. View real-time results in the application window

## Directory Structure

```
project/
├── yolov5-7.0/
│   ├── detect.py         # Script 1: YOLOv5 detection
│   └── data/images/      # Storage for captured images
├── VGG16/
│   ├── crop.py           # Script 2: Crops detected screws
│   ├── predictdata/      # Directory for cropped screw images
│   └── VGG16_train/
│       └── VGG16Predict.py  # Script 3: Condition classification
└── main.py               # Main application with GUI
```

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- Pillow
- Tkinter

## Future Developments

- Support for additional screw types and conditions
- Performance optimization for edge devices
- Integration with industrial automation systems
- Expanded analytics and reporting capabilities
