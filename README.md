 # Traffic Congestion Control Using YOLOv8

This project aims to use **YOLOv8** (You Only Look Once version 8) for detecting and analyzing traffic congestion in real-time. By detecting various vehicles on the road, the system helps in controlling and monitoring traffic congestion more efficiently.

## Features

- **Real-time Traffic Detection:** Detects vehicles and counts them to determine traffic congestion levels.
- **Vehicle Classification:** Classifies different types of vehicles (e.g., cars, bikes, trucks, etc.) based on detected objects.
- **Congestion Analysis:** Analyzes the vehicle count to evaluate the level of congestion in different areas.
- **YOLOv8 Object Detection:** Utilizes the YOLOv8 model for fast and accurate object detection in video footage.

## Technologies Used

- **YOLOv8:** Object detection model for real-time detection of vehicles.
- **Python:** Primary programming language used for the project.
- **OpenCV:** For real-time video processing and handling.
- **Git Large File Storage (LFS):** For handling large files like the trained YOLO weights and video datasets.
- **TensorFlow/PyTorch (Optional):** For handling model training and fine-tuning.

## Installation

### Prerequisites

1. Python 3.8+
2. Git
3. Git LFS (for large files)
4. YOLOv8 pre-trained weights
5. requirements.txt

### Clone the Repository

```bash
git clone https://github.com/iamsahilkansal/Traffic-Congestion-Control-Using-YOLOv8.git
cd Traffic-Congestion-Control-Using-YOLOv8
```

### Install Required Dependencies
```bash
pip install -r requirements.txt
```
### Output
![Traffic-Congestion-Control-Using-YOLOv8](assets/img1.png)

### Key Points to Note:
- The instructions provided in this `README.md` cover the repository setup, installation, usage, and structure.
- Replace `detect_traffic.py` with the actual filename if it's different in your project.
- Adjust paths or steps depending on your actual project structure.
- This format uses standard Markdown for GitHub.

Let me know if you need any adjustments!
