import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load NPU delegate
delegate = tflite.load_delegate("libvx_delegate.so")

# Load TFLite model with NPU delegate
interpreter = tflite.Interpreter(
    model_path="mobilenet_v1_1.0_224_quant.tflite",
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
img = Image.open("input.jpg").resize((224, 224))
img = np.array(img).astype(np.uint8)
img = np.expand_dims(img, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

# Get output and print top-1 prediction
output = interpreter.get_tensor(output_details[0]['index'])
label_id = np.argmax(output)
labels = open("imagenet_labels.txt").read().splitlines()
print("Prediction:", labels[label_id])
