# Auralith AR Pipeline — CNN-Based (Iteration 1)
**Author: Pirassena Sabaratnam**

## Overview
This repository contains the first iteration of the Auralith AR ecosystem, built for the NAIMME live-shopping app. It implements a **real-time AR clothing try-on pipeline** using standard CNN backbones (MobileNetV2/EfficientNet).

## Architecture
The pipeline consists of three modules:

### 1. FLUXA — Multi-Task Perception
Trained on COCO2017 for human subject analysis. Outputs four task heads simultaneously:
- **Semantic Segmentation**: Subject/background masking
- **Keypoint Detection**: 17 skeletal joint positions for pose tracking
- **Surface Normals**: Per-pixel 3D surface orientation for garment draping
- **Environment Lighting**: 9-channel spherical harmonic lighting estimation

### 2. LITHOS — Garment Encoder
Encodes target garment images into a latent representation, preserving visual detail independent of the target environment.

### 3. PRISM — Synthesis & Rendering
Merges FLUXA's perception outputs with LITHOS's garment encoding to render virtual clothing onto the subject:
- Pose-driven garment deformation
- Shadow synthesis from estimated geometry
- Dynamic relighting using FLUXA's environment lighting

## Technical Details
- **Framework**: TensorFlow 2.x / Keras
- **Backbones**: MobileNetV2 / EfficientNet (configurable)
- **Deployment**: Google Cloud Vertex AI training, TFLite export for on-device inference
- **Dataset**: COCO2017 (human subjects)

## License
Apache License 2.0 — see [LICENSE](LICENSE).

---
*Developed March 2025 — Auralith Inc.*
