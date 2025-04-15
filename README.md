# Auralith AR Ecosystem (Iteration 1: CNN-Based)
**Author: Pirassena Sabaratnam**

## Overview
This repository contains the first full-scale iteration of the Auralith Augmented Reality (AR) ecosystem, developed in early 2025 specifically for the **NAIMME app**. This iteration utilizes a standard Convolutional Neural Network (CNN) architecture (MobileNetV2/EfficientNet backbones) to achieve **Real-Time AR Clothing Try-On**.

While modular and adaptable to various use cases, the primary implementation focuses on the seamless integration of garment encoding and live subject perception.

## System Anatomy (AR Try-On Pipeline)
The ecosystem is divided into three core specialized modules that work in concert:

### 1. FLUXA (Perception Engine)
**Role: Subject Identification**
Trained on COCO2017 for human subject analysis, FLUXA performs real-time perception of the user:
- **Semantic Segmentation**: Isolating the subject from the background.
- **Keypoint Detection**: Tracking skeletal joints to align virtual garments.
- **Surface Normals**: Estimating the user's 3D geometry for realistic garment draping.
- **Environment Lighting**: Inferring global illumination to relight virtual clothing.

### 2. LITHOS (Scene & Garment Encoding)
**Role: Garment Representation**
Trained on specialized clothing datasets, LITHOS encodes the target garment into a consistent latent representation, ensuring the visual details are preserved regardless of the final environment.

### 3. PRISM (Augmentation Engine)
**Role: Final Synthesis**
The decoding and synthesis layer that merges the perception data from FLUXA with the garment encoding from LITHOS to place the virtual clothing onto the subject in real-time.
- **Deformation**: Morphing the virtual garment to match the subject's pose.
- **Shadow Synthesis**: Generating physically-aligned shadows for realistic depth.
- **Lighting Application**: Dynamically relighting the garment using FLUXA's environment data.

## Technical Implementation
- **Framework**: TensorFlow 2.x / Keras
- **Deployment**: Optimized for **Google Cloud Vertex AI** with GCS-integrated data pipelines.
- **Mobile Support**: Fully compatible with **TFLite** for on-device AR execution.
- **Optimization**: Features custom callbacks for GCS checkpointing, automated metric tracking, and early overfitting detection.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---
*Developed in 2025 as part of the Auralith Inc. Research.*
