# OP_FractureScope
"Operation FractureScope" – Industrial Anomaly Detection in Harsh Environments
**FractureScope** is an AI-powered visual inspection framework for detecting surface anomalies—such as cracks, corrosion, and leaks—on pipelines or industrial surfaces. It leverages multimodal early fusion (RGB + thermal), edge deployment with ONNX, and secure logging using blockchain concepts for compliance and traceability.

---

## 🔍 Project Objectives

- Detect surface anomalies using early fusion of RGB and thermal images.
- Optimize model for **real-time edge inference** (Jetson Nano/Orin, Raspberry Pi).
- Enable **secure and tamper-proof logging** of inspection metadata via blockchain.
- Address data scarcity using **synthetic data** and **federated learning** strategies.
- Integrate feedback loop over **4G connectivity** between robot and operator.

---

## 📁 Directory Structure & File Descriptions

```
OP_FractureScope
├── data // contains all images and datasets
├── deploy // contains all quantization, pruning, onnx and edge inference scripts
├── outputs //contains all runs outputs
├── requirements.txt // list of dependencies
├── report // contains PDF report describing modeling, design, and conclusions 
├── src // contains all model definition and training related scripts
└── README.md

data
├── sample_images // images used for edge inference using onnxruntime
├── sdnet // Original SDNET2018 images used to synthesize our dataset
│   ├── cracked
│   └── non_cracked
├── synth // Our Main Dataset
│   ├── corrosion // CLASS: 1
│   │   ├── thermal  
│   │   └── visual
│   ├── cracked // CLASS: 0
│   │   ├── thermal
│   │   └── visual
│   ├── leak // CLASS: 2
│   │   ├── thermal
│   │   └── visual
│   └── test_dataset // Test Dataset synthesized for model evaluation
│       ├── corrosion
│       │   ├── thermal
│       │   └── visual
│       ├── cracked
│       │   ├── thermal
│       │   └── visual
│       └── leak
│             ├── thermal
│             └── visual
└── textures
	└── rust_texture.png // reference rust texture used for synthesis of “corrosion’ images

src
├── data
│   ├── dataset.py // file with Custom PyTorch Dataset class SynthDataset and Loader
│   └── transforms.py // data augmentations script
├── models
│   └── early_fusion.py // our EarlyFusionCNN model definition
├── train
│   ├── evaluate_best_model.py // model evaluation script
│   └── train.py // main training loop
└── utils
    ├── generate_test_thermal.py // script to synthesize thermal images for TEST dataset
    ├── generate_thermal.py // script to synthesize thermal images
    ├── simulate_defects.py // script for synthesis of defects, “corrosion” and “leak”
    └── simulate_test_dataset.py // script to generate the TEST dataset

outputs
├── metrics // contains all output metrices and plots after training
│   ├── accuracy_plot_20250505-060834.png
│   ├── classification_report_20250505-060834.txt
│   ├── confusion_matrix_20250505-060834.png
│   └── loss_plot_20250505-060834.png
├── models // saved model checkpoints
│   ├── best_acc_model_20250505-060834.pth
│   └── early_fusion_best_loss20250505-060834.pth
├── tensorboard // tensorboard logs
│   └── 20250505-060834
└── test_results // model evaluation results on TEST dataset
    ├── classification_report.csv
    ├── confusion_matrix.png
    └── evaluation_summary.txt

deploy
├── edge_inference_demo.py // script of running edge inference on onnxruntime
├── edge_inference_visualizer.py // script of edge inference on onnxruntime (with visual output)
├── optim_models // contans all saved optimized models
│   ├── onnx
│   │   ├── EarlyFusionCNN_pruned.onnx // pruned model
│   │   └── EarlyFusionCNN_pruned_quant.onnx // pruned, qunatized and onnx exported final model
│   └── optimizedEarlyFusionCNN.pth 
├── optimize_model.py //script to apply pruning and quantization only
├── prune_onnx_export.py // script to apply pruning and onnx export
└── quantize_onnx_export.py // final script to apply pruning, quantization and exoprt to onnx


```

---

## 🧠 Model

- Architecture: `EarlyFusionCNN` with 6-channel input (RGB + simulated thermal).
- Classes: `cracked`, `corrosion`, `leak`
- Loss: Categorical CrossEntropy
- Optimizations: Pruning + Quantization + ONNX Export

---

## 🚀 Setup Instructions

1. **Create Environment**
   ```bash
   python3 -m venv edgex_mpd
   source edgex_mpd/bin/activate
   pip install -r requirements.txt
  
📘 Report
See report/Report_FractureScope_MayurPD.pdf for full technical details, methodology, results, and system architecture.

👨‍🔧 Authors
Mayur [Project Lead & Developer]
