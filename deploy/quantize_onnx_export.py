from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import onnx
import os
import numpy as np
import cv2

# Configuration
model_fp32 = "deploy/optim_models/onnx/EarlyFusionCNN_pruned.onnx"
model_quant = "deploy/optim_models/onnx/EarlyFusionCNN_pruned_quant.onnx"
calib_dir = "data/sample_images"  # Folder with a few sample 6-channel images

# Dummy Calibration Reader
class ImageCalibReader(CalibrationDataReader):
    def __init__(self, image_folder):
        self.image_folder = sorted(os.listdir(image_folder))
        self.image_folder = [os.path.join(image_folder, f) for f in self.image_folder if f.endswith(".jpg") or f.endswith(".png")]
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            inputs = []
            for path in self.image_folder:
                img = cv2.imread(path)  # Load 3-channel image
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0
                # Duplicate channels to simulate 6-channel input
                img_fused = np.concatenate([img, img], axis=2).transpose(2, 0, 1)  # (6, H, W)
                inputs.append({"input": np.expand_dims(img_fused, axis=0)})
            self.enum_data = iter(inputs)
        return next(self.enum_data, None)

# Run Quantization
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

quantize_static(
    model_fp32,
    model_quant,
    calibration_data_reader=ImageCalibReader(calib_dir),
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8
)

print(f"Quantized model saved to: {model_quant}")
