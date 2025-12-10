🚀 NPU Live Streaming Project

This project demonstrates live object detection using a MobileNet TFLite model accelerated by an NPU on an embedded board (e.g., i.MX8M Plus). It captures video from a camera, runs inference via NPU, and streams results through gstreamer.

## 📂 Step 1: Setup Directory
```bash
ssh user@<board_ip>
mkdir ~/npu_test
cd ~/npu_test
```
## 🛠 Step 2: Install Dependencies

Update packages:
```bash
sudo apt update && sudo apt upgrade -y
```

Install OpenGL libraries:
```bash
sudo apt install -y libgl1-mesa-glx libglu1-mesa libopengl0
```

Install Python packages:
```bash
pip3 install --user flask opencv-python-headless numpy tflite-runtime
```

## ⚠️ Tip: If OpenCV fails due to NumPy 2.x:
```bash
pip3 install --user "numpy<2"
```
## 📹 Step 3: Camera Setup

List cameras:
```bash
ls /dev/video*
```

Test camera indices:
```bash
import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    print(f"Camera {i}: {'Working' if ret else 'Failed'}")
    cap.release()
```

Use the working index in live_gstream_object_detection.py: 
```bash
cap = cv2.VideoCapture(<working_index>)
```
## 📝 Step 4: Add Project Files

```
live_gstream_object_detection.py  – Main Flask live streaming script

mobilenet_v1_1.0_224_quant.tflite – TFLite model

imagenet_labels.txt – Labels for classification

test_npu.py - For one image detection
```

## ▶️ Step 5: Run Live Stream
```bash
python3  live_gstream_object_detection.py
```

You should see live camera feed with bounding boxes and labels.

## ⚠️ Troubleshooting

-  Camera not detected: Verify /dev/videoX index and test individually.

- OpenCV import fails: Ensure OpenGL libs installed & compatible NumPy.

- TFLite/NPU errors: Verify delegate path in live_stream.py (/usr/lib/libvx_delegate.so).
