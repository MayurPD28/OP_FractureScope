import onnxruntime as ort
import numpy as np
import os
import cv2

# Config
model_path = "deploy/optim_models/onnx/EarlyFusionCNN_pruned_quant.onnx"
image_folder = "data/sample_images"  # Folder with RGB images
input_size = (224, 224)

# Load model
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Inference loop
for filename in sorted(os.listdir(image_folder)):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0

    # Simulate 6-channel early fusion (RGB + RGB Thermal)
    fused_img = np.concatenate([img, img], axis=2).transpose(2, 0, 1)  # (6, 224, 224)
    fused_img = np.expand_dims(fused_img, axis=0)  # (1, 6, 224, 224)

    # Run inference
    outputs = session.run(None, {input_name: fused_img})
    prediction = np.argmax(outputs[0], axis=1)[0]

    labels = {0: "crack", 1: "corrosion", 2: "leak"}
    print(f"[{filename}] → Predicted Class: {labels[prediction]}")
    # print(f"[{filename}] → Predicted Class: {prediction}")
