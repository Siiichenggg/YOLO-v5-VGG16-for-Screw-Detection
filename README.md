# YOLO-v5-VGG16-for-Screw-Detection

**Project Overview**

This project combines YOLOv5 and VGG16 models to achieve screw detection and condition recognition. First, the YOLOv5 model is used to detect the location of screws in an image. Then, the VGG16 model is used to determine whether the screws are rusted or slipped.

**Key Features**

*   **Screw Detection (YOLOv5):**
    *   Utilizes a YOLOv5 model, trained on a screw dataset, to accurately detect the position of screws in images.
*   **Screw Condition Recognition (VGG16):**
    *   Employs a VGG16 model, trained on a dataset of rusted and slipped screws, to distinguish whether a screw is rusted ("marked as Chinese Pinyin: shengxiu") or slipped ("marked as Chinese Pinyin: huasi").
*   **Real-time Detection (main.py):**
    *   The `main.py` file creates a Graphical User Interface (GUI) and accesses the MacBook's camera to enable the test of real-time screw detection and condition recognition.

**Technology Stack**

*   **YOLOv5:** Used for object detection.
*   **VGG16:** Used for image classification (screw condition recognition).
*   **Python:** Primary programming language.
*   **PyTorch:** Deep learning framework.

I hope this optimized README file is helpful!
