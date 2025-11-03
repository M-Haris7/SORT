# Multi-Object Tracking with YOLO and SORT

A Python implementation of Multi-Object Tracking (MOT) using YOLOv8 for object detection and SORT (Simple Online and Realtime Tracking) algorithm for tracking.

## 📋 Project Overview

This project implements a tracking-by-detection approach for multi-object tracking in video sequences. It combines:
- **YOLOv8** for real-time object detection
- **SORT Algorithm** with Kalman filtering for object tracking
- **MOT Metrics** for performance evaluation

**Assignment:** AI/ML and Applications (MDS 302) - Assignment 3  
**Dataset:** MOT15 Benchmark (ETH-Bahnhof Sequence)

---

## 🎯 Features

- ✅ Real-time object detection using YOLOv8
- ✅ Multi-object tracking with SORT algorithm
- ✅ Kalman filter for motion prediction
- ✅ Hungarian algorithm for data association
- ✅ Video output with bounding boxes and track IDs
- ✅ MOT format output file generation
- ✅ Automatic evaluation with MOTA and MOTP metrics

---

## 📊 Results

### Performance Metrics on MOT15 (ETH-Bahnhof)

| Metric | Value | Description |
|--------|-------|-------------|
| **MOTA** | **63.68%** | Multiple Object Tracking Accuracy |
| **MOTP** | **0.8511** | Multiple Object Tracking Precision |
| Frames Processed | 1000 | Total frames |
| Successful Matches | 5030 | Correct detections |
| False Positives | 146 | Incorrect detections |
| False Negatives | 2439 | Missed detections |
| ID Switches | 201 | Identity changes |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download this repository**
```bash
cd MOT_Assignment
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python
- numpy
- scipy
- filterpy
- ultralytics
- motmetrics

### Dataset Setup

1. **Download MOT15 dataset**
   - Visit: https://motchallenge.net/data/MOT15/
   - Download the training data
   - Extract to project folder

2. **Verify folder structure**
```
MOT_Assignment/
├── MOT15/
│   └── train/
│       └── ETH-Bahnhof/
│           ├── img1/           # Video frames
│           └── gt/
│               └── gt.txt      # Ground truth
├── mot_tracker.py
└── requirements.txt
```

---

## 💻 Usage

### Basic Usage

Run the tracker with default settings:

```bash
python mot_tracker.py
```

### Advanced Usage

Modify parameters in the code:

```python
# In mot_tracker.py, modify these parameters:

# YOLO detection confidence
detection_confidence = 0.5  # Range: 0.3 - 0.7

# SORT parameters
tracker = Sort(
    max_age=1,           # Frames to keep track alive (1-5)
    min_hits=3,          # Detections before confirmation (1-5)
    iou_threshold=0.3    # IOU threshold for matching (0.2-0.5)
)
```

### Running on Different Sequences

```python
# Change sequence path
sequence_path = "MOT15/train/ETH-Bahnhof"  # Default
# sequence_path = "MOT15/train/Venice-2"   # Alternative
# sequence_path = "MOT15/train/KITTI-17"   # Alternative
```

---

## 📁 Output

After running, the following files are generated in `output/` folder:

1. **output_video.mp4**
   - Video with tracked objects
   - Colored bounding boxes
   - Track ID labels

2. **tracking_results.txt**
   - MOT format output
   - Format: `frame_id, track_id, x, y, width, height, conf, -1, -1, -1`

Example output:
```
1,1,794.5,247.3,71.2,174.5,1,-1,-1,-1
1,2,1648.1,385.7,99.3,234.2,1,-1,-1,-1
2,1,795.3,248.1,71.3,173.8,1,-1,-1,-1
```

---

## 🔧 Project Structure

```
MOT_Assignment/
│
├── MOT15/                      # Dataset directory
│   └── train/
│       └── ETH-Bahnhof/
│           ├── img1/          # Video frames
│           └── gt/
│               └── gt.txt     # Ground truth
│
├── mot_tracker.py             # Main implementation
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
└── output/                    # Generated outputs
    ├── output_video.mp4      # Tracked video
    └── tracking_results.txt  # MOT format results
```

---

## 🧠 Algorithm Details

### SORT Algorithm

SORT (Simple Online and Realtime Tracking) consists of:

1. **Detection**: YOLOv8 detects objects in each frame
2. **Prediction**: Kalman filter predicts next position
3. **Association**: Hungarian algorithm matches detections to tracks
4. **Update**: Matched tracks updated with new detections
5. **Management**: Create new tracks, remove old ones

### Kalman Filter

- **State Vector**: `[x, y, s, r, vx, vy, vs]`
  - x, y: Center position
  - s: Scale (area)
  - r: Aspect ratio
  - vx, vy, vs: Velocities

- **Motion Model**: Constant velocity

### Data Association

- **Method**: Hungarian algorithm
- **Cost Metric**: 1 - IOU (Intersection over Union)
- **Threshold**: IOU > 0.3 for valid match

---

## 📈 Evaluation Metrics

### MOTA (Multiple Object Tracking Accuracy)

Measures overall tracking accuracy considering:
- False Negatives (missed detections)
- False Positives (incorrect detections)
- ID Switches (identity changes)

**Formula:**
```
MOTA = 1 - (FN + FP + IDSW) / GT
```

**Interpretation:**
- > 70%: Excellent
- 50-70%: Good ✓ (Our result: 63.68%)
- 30-50%: Average
- < 30%: Poor

### MOTP (Multiple Object Tracking Precision)

Measures spatial accuracy of bounding boxes.

**Formula:**
```
MOTP = Average IOU of all matched objects
```

**Interpretation:**
- > 0.80: Excellent ✓ (Our result: 0.8511)
- 0.70-0.80: Good
- 0.60-0.70: Average
- < 0.60: Poor

---

## ⚙️ Parameter Tuning Guide

### Detection Confidence Threshold

```python
detection_confidence = 0.5  # Default
```

- **Lower (0.3-0.4)**: More detections, more false positives
- **Higher (0.6-0.7)**: Fewer false positives, more missed detections

### max_age

```python
max_age = 1  # Default
```

- **Lower (1)**: Removes lost tracks quickly, fewer false tracks
- **Higher (3-5)**: Keeps tracks longer, better for occlusions

### min_hits

```python
min_hits = 3  # Default
```

- **Lower (1-2)**: Confirms tracks faster, more false positives
- **Higher (4-5)**: More reliable tracks, slower confirmation

### iou_threshold

```python
iou_threshold = 0.3  # Default
```

- **Lower (0.2)**: More lenient matching, fewer ID switches
- **Higher (0.4-0.5)**: Stricter matching, more ID switches

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Module not found error**
```bash
# Solution: Install missing package
pip install <package_name>
```

**Issue 2: Dataset not found**
```bash
# Solution: Check folder structure and paths
# Ensure MOT15/train/ETH-Bahnhof/ exists
```

**Issue 3: YOLO model download fails**
```bash
# Solution: Check internet connection
# Model auto-downloads on first run (~6MB)
```

**Issue 4: Out of memory**
```bash
# Solution: Use smaller YOLO model
# Change: YOLO('yolov8n.pt')  # nano version
```

**Issue 5: Slow processing**
```bash
# Solution: 
# - Use GPU if available
# - Reduce video resolution
# - Use yolov8n (nano) instead of larger models
```

---

## 🔄 Extending the Project

### Use Different YOLO Models

```python
# Faster but less accurate
self.model = YOLO('yolov8n.pt')  # Nano (default)

# More accurate but slower
self.model = YOLO('yolov8s.pt')  # Small
self.model = YOLO('yolov8m.pt')  # Medium
self.model = YOLO('yolov8l.pt')  # Large
self.model = YOLO('yolov8x.pt')  # Extra Large
```

### Track Different Object Classes

```python
# Current: Only persons (class 0)
if box.cls[0] == 0:  # Person

# Track cars (class 2)
if box.cls[0] == 2:  # Car

# Track multiple classes
if box.cls[0] in [0, 2, 7]:  # Person, Car, Truck
```

### Upgrade to Deep SORT

Add appearance features for better re-identification:
- Extract CNN features from detected objects
- Use cosine distance for appearance matching
- Combine with motion information

---

## 📚 References

1. Bewley, A., et al. (2016). "Simple online and realtime tracking." *ICIP 2016*
2. Redmon, J., & Farhadi, A. (2018). "YOLOv3: An Incremental Improvement"
3. MOT Challenge: https://motchallenge.net/
4. Ultralytics YOLOv8: https://docs.ultralytics.com/

---


---

## 📄 License

This project is for educational purposes as part of the AI/ML and Applications course (MDS 302).

---


## 🤝 Acknowledgments

- MOT Challenge for providing the benchmark dataset
- Alex Bewley for the original SORT implementation
- Ultralytics team for YOLOv8
- Course instructors and teaching assistants



**Last Updated:** November 3, 2025

**Status:** ✅ Complete and Ready for Submission
