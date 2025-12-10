import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# ------------------------------
# Camera Setup (/dev/video4)
# ------------------------------
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("❌ Failed to open /dev/video4")

# ------------------------------
# Load NPU TFLite Model
# ------------------------------
delegate = tflite.load_delegate("/usr/lib/libvx_delegate.so")

interpreter = tflite.Interpreter(
    model_path="mobilenet_v1_1.0_224_quant.tflite",
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("imagenet_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ------------------------------
# Main Loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Preprocess for model input (224x224)
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img.astype(np.uint8), axis=0)

    # Run NPU inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    label = labels[int(np.argmax(output))]

    # Draw rectangle + label on original frame
    h, w, _ = frame.shape
    box_size = min(h, w) // 3  # auto size rectangle
    top_left = (w//2 - box_size//2, h//2 - box_size//2)
    bottom_right = (w//2 + box_size//2, h//2 + box_size//2)

    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(frame, label, (top_left[0], top_left[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show live output
    cv2.imshow("NPU Object Detection (/dev/video4)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# Cleanup
# ------------------------------
cap.release()
cv2.destroyAllWindows()

