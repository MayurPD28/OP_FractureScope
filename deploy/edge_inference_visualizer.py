import onnxruntime as ort
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Config
model_path = "deploy/optim_models/onnx/EarlyFusionCNN_pruned_quant.onnx"
image_folder = "data/sample_images"  # Folder with RGB images
input_size = (224, 224)

# Load model
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Labels mapping
labels = {0: "cracked", 1: "corrosion", 2: "leak"}

# Inference loop
for filename in sorted(os.listdir(image_folder)):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path)
    original_img = img.copy()  # Save original image for visualization

    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0

    # Simulate 6-channel early fusion (RGB + RGB Thermal)
    fused_img = np.concatenate([img, img], axis=2).transpose(2, 0, 1)  # (6, 224, 224)
    fused_img = np.expand_dims(fused_img, axis=0)  # (1, 6, 224, 224)

    # Run inference
    outputs = session.run(None, {input_name: fused_img})
    prediction = np.argmax(outputs[0], axis=1)[0]
    predicted_class = labels[prediction]

    # Draw the predicted class on the image
    text = f"Predicted: {predicted_class}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)  # Green text
    thickness = 2

    # Put text on the image
    cv2.putText(original_img, text, (10, 30), font, 1, color, thickness, cv2.LINE_AA)

    # Convert BGR to RGB for matplotlib display
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Show image with prediction using matplotlib
    plt.imshow(original_img_rgb)
    plt.title(f"Prediction: {predicted_class}")
    plt.axis('off')  # Hide axes
    plt.show()  # Display the image

    print(f"[{filename}] → Predicted Class: {predicted_class}")
