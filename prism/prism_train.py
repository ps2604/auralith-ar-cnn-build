#!/usr/bin/env python3
"""
Improved PRISM Module Training Script for Vertex AI

PRISM is the renderer component of Auralith that combines:
- Environmental data from FLUXA (pose detection, lighting, surface normals)
- Base encoding from LITHOS (material and property information)

This improved version:
1. Fixes numerical stability issues that cause NaN values
2. Simplifies architecture for more reliable training
3. Improves gradient flow with better normalization
4. Adds extensive error checking and debugging
5. Uses proper tensor dtype handling for mixed precision

Author: Pirassena Sabaratnam
Date: May 2025
"""

import os
import sys
import time
import io
import json
import random
import logging
import argparse
import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime
from pathlib import Path
from PIL import Image
from google.cloud import storage
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure TensorFlow for better numerical stability
tf.keras.backend.set_floatx('float32')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TensorFlow imports
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Dropout, 
    UpSampling2D, Concatenate, Dense, GlobalAveragePooling2D, 
    Layer, Add, Lambda, Reshape, Multiply, AveragePooling2D,
    LeakyReLU, LayerNormalization, SeparableConv2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError

# Constants for keypoint map shape
KEYPOINT_H = 30
KEYPOINT_W = 32
NUM_KEYPOINTS = 17

# =========================
# Command Line Argument Setup
# =========================

def parse_args():
    """Parse command line arguments for the PRISM training script."""
    parser = argparse.ArgumentParser(description='Improved PRISM Training Script for Auralith')
    
    # Core cloud arguments
    parser.add_argument('--project-id', type=str, default='bright-link-455716-h0')
    parser.add_argument('--bucket-name', type=str, default='auralith')
    parser.add_argument('--base-path', type=str, default='prism')
    
    # Input paths
    parser.add_argument('--fluxa-data-path', type=str, default='fluxa_data')
    parser.add_argument('--lithos-encoding-path', type=str, default='lithos_encodings')
    parser.add_argument('--person-images-path', type=str, default='person_images')
    parser.add_argument('--clothing-images-path', type=str, default='clothing_images')
    parser.add_argument('--product-metadata-path', type=str, default='product_metadata')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--val-batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--initial-epoch', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=0.0002)
    parser.add_argument('--checkpoint-steps', type=int, default=500)
    
    # Data limits
    parser.add_argument('--max-samples', type=int, default=100000)
    parser.add_argument('--max-val-samples', type=int, default=10000)
    
    # Model configuration
    parser.add_argument('--input-size', type=str, default="768,768", 
                       help="Height,width of input images")
    parser.add_argument('--lithos-type', type=str, default='direct',
                        choices=['direct', 'dialect'],
                        help="LITHOS encoding type: direct (768x768) or dialect (1024x1024)")
    parser.add_argument('--fse-channels', type=int, default=6, 
                        help="Number of FSE channels in LITHOS encoding")
    parser.add_argument('--use-attention', action='store_true',
                        help="Enable attention mechanisms in the model")
    parser.add_argument('--use-transformer', action='store_true',
                        help="Use transformer blocks instead of convolutional blocks")
    
    # Regularization
    parser.add_argument('--dropout-rate', type=float, default=0.1)
    parser.add_argument('--l2-reg', type=float, default=1e-6)
    
    # Loss weights
    parser.add_argument('--clothing-weight', type=float, default=2.0,
                        help="Weight for clothing regions in the loss function")
    parser.add_argument('--background-weight', type=float, default=0.5,
                        help="Weight for background regions in the loss function")
    parser.add_argument('--perceptual-weight', type=float, default=0.1,
                        help="Weight for perceptual loss component")
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help="Value for gradient clipping")
    
    # Training controls
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--use-validation', action='store_true')
    parser.add_argument('--skip-checkpoint', action='store_true')
    parser.add_argument('--debug-visualize', action='store_true',
                        help="Generate debug visualizations during training")
    parser.add_argument('--local-tmp-dir', type=str, default='/tmp/auralith')
    
    # Checkpoint loading
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help="Path to specific checkpoint to load")
    parser.add_argument('--load-step', type=int, default=None,
                        help="Load checkpoint at specific step")
    
    # Mixed precision
    parser.add_argument('--mixed-precision', action='store_true',
                        help="Enable mixed precision training")
    
    # Distributed training
    parser.add_argument('--multi-gpu', action='store_true',
                        help="Enable multi-GPU training with MirroredStrategy")
    
    # Debugging options
    parser.add_argument('--debug-nan', action='store_true',
                        help="Enable NaN debugging with tf.debugging.enable_check_numerics")
    parser.add_argument('--simple-model', action='store_true',
                        help="Use a simplified model for debugging")
    # Add these to the existing argument parser
    parser.add_argument('--render-targets-path', type=str, default='targets/render_targets',
                        help="Path to render target images")
    parser.add_argument('--generate-triplets', action='store_true',
                        help="Generate new training triplets CSV")
    parser.add_argument('--triplets-per-product', type=int, default=5,
                        help="Number of persons to assign to each product when generating triplets")
    parser.add_argument('--triplets-csv-path', type=str, default='train_triplets.csv',
                        help="Path to CSV file containing training triplets")
    return parser.parse_args()

# Parse arguments globally
args = parse_args()

# Enable TensorFlow debugging if requested
if args.debug_nan:
    tf.debugging.enable_check_numerics()
    logger.info("🔍 NaN checking enabled - this will slow down training")

# =========================
# Global Configuration Setup
# =========================

# Parse input size
try:
    INPUT_HEIGHT, INPUT_WIDTH = map(int, args.input_size.split(','))
except:
    logger.warning(f"⚠️ Invalid input size format: {args.input_size}, using default 768x768")
    INPUT_HEIGHT, INPUT_WIDTH = 768, 768

# Configure LITHOS encoding parameters based on type
if args.lithos_type == 'direct':
    LITHOS_HEIGHT, LITHOS_WIDTH = INPUT_HEIGHT, INPUT_WIDTH  # Match input size
    LITHOS_CHANNELS = args.fse_channels
elif args.lithos_type == 'dialect':
    LITHOS_HEIGHT, LITHOS_WIDTH = 1024, 1024  # High-resolution dialect encoding
    LITHOS_CHANNELS = args.fse_channels
else:
    logger.warning(f"⚠️ Unknown LITHOS type: {args.lithos_type}, defaulting to direct")
    LITHOS_HEIGHT, LITHOS_WIDTH = INPUT_HEIGHT, INPUT_WIDTH
    LITHOS_CHANNELS = args.fse_channels

# GCS paths
PROJECT_ID = args.project_id
GCS_BUCKET_NAME = args.bucket_name
GCS_BASE_PATH = args.base_path
LOCAL_TEMP_DIR = args.local_tmp_dir

# Construct full GCS paths
GCS_FLUXA_DATA_PATH = f"{GCS_BASE_PATH}/{args.fluxa_data_path}"
GCS_LITHOS_ENCODING_PATH = f"{GCS_BASE_PATH}/{args.lithos_encoding_path}"
GCS_PERSON_IMAGES_PATH = f"{GCS_BASE_PATH}/{args.person_images_path}"
GCS_CLOTHING_IMAGES_PATH = f"{GCS_BASE_PATH}/{args.clothing_images_path}"
GCS_PRODUCT_METADATA_PATH = f"{GCS_BASE_PATH}/{args.product_metadata_path}"
GCS_CHECKPOINT_DIR = f"{GCS_BASE_PATH}/checkpoints"
GCS_OUTPUT_DIR = f"{GCS_BASE_PATH}/outputs"
GCS_LOG_DIR = f"{GCS_BASE_PATH}/logs"
GCS_RENDER_TARGETS_PATH = f"{GCS_BASE_PATH}/{args.render_targets_path}"

# Validation paths
GCS_VAL_FLUXA_DATA_PATH = f"{GCS_BASE_PATH}/val/{args.fluxa_data_path}"
GCS_VAL_LITHOS_ENCODING_PATH = f"{GCS_BASE_PATH}/val/{args.lithos_encoding_path}"
GCS_VAL_PERSON_IMAGES_PATH = f"{GCS_BASE_PATH}/val/{args.person_images_path}"
GCS_VAL_CLOTHING_IMAGES_PATH = f"{GCS_BASE_PATH}/val/{args.clothing_images_path}"
GCS_VAL_PRODUCT_METADATA_PATH = f"{GCS_BASE_PATH}/val/{args.product_metadata_path}"
GCS_VAL_RENDER_TARGETS_PATH = f"{GCS_BASE_PATH}/val/{args.render_targets_path}"

# Create local temp directories
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
os.makedirs(os.path.join(LOCAL_TEMP_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(LOCAL_TEMP_DIR, "visualizations"), exist_ok=True)
os.makedirs(os.path.join(LOCAL_TEMP_DIR, "logs"), exist_ok=True)

# Connect to Google Cloud Storage
try:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    logger.info(f"✅ Connected to GCS bucket '{GCS_BUCKET_NAME}'")
except Exception as e:
    logger.error(f"❌ Failed to connect to GCS: {e}")
    sys.exit(1)

# Configure mixed precision if enabled
if args.mixed_precision:
    logger.info("🔧 Enabling mixed precision training")
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

# Configure distributed training if enabled
if args.multi_gpu:
    try:
        logger.info("🔄 Setting up distributed training")
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"✅ Using {strategy.num_replicas_in_sync} devices for distributed training")
    except Exception as e:
        logger.error(f"❌ Failed to set up distributed training: {e}")
        strategy = None
else:
    strategy = None

# =========================
# File and Dataset Utilities
# =========================

def list_available_files(prefix_path, file_extension='.npz'):
    """List available files with matching extension in a GCS prefix path."""
    try:
        logger.info(f"Searching in: {prefix_path}")
        
        blobs = list(bucket.list_blobs(prefix=prefix_path, delimiter='/'))
        files = [b.name for b in blobs if b.name.endswith(file_extension)]
        
        logger.info(f"Found {len(files)} files ending with {file_extension} in {prefix_path}")
        if files:
            logger.info(f"Sample files: {files[:5]}")
        
        return files
    except Exception as e:
        logger.error(f"❌ Error listing files from {prefix_path}: {e}")
        return []

def generate_training_triplets_csv():
    """
    Generate a CSV file with product ID, person ID, and render target ID.
    Modified to use render targets as both person images and reference images.
    """
    logger.info("📊 Generating new training triplets CSV...")
    
    # Check LITHOS encodings (.npz files)
    logger.info(f"Searching for LITHOS encodings in: {GCS_LITHOS_ENCODING_PATH}")
    lithos_blobs = list(bucket.list_blobs(prefix=GCS_LITHOS_ENCODING_PATH))
    lithos_ids = set()
    
    for blob in lithos_blobs:
        if blob.name.endswith('.npz'):
            product_id = os.path.splitext(os.path.basename(blob.name))[0]
            lithos_ids.add(product_id)
    
    logger.info(f"Found {len(lithos_ids)} LITHOS encodings")
    
    # Check render targets (.jpg, .png, .jpeg files)
    logger.info(f"Searching for render targets in: {GCS_RENDER_TARGETS_PATH}")
    render_blobs = list(bucket.list_blobs(prefix=GCS_RENDER_TARGETS_PATH))
    render_ids = set()
    
    for blob in render_blobs:
        if any(blob.name.endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):
            render_id = os.path.splitext(os.path.basename(blob.name))[0]
            render_ids.add(render_id)
    
    logger.info(f"Found {len(render_ids)} render targets")
    
    # Find products with both LITHOS encodings and render targets
    product_ids_with_renders = lithos_ids.intersection(render_ids)
    
    if not product_ids_with_renders:
        logger.warning("⚠️ No product IDs with both LITHOS encodings and render targets found!")
        # Print some example IDs to help debug
        if lithos_ids:
            logger.info(f"Sample LITHOS IDs: {list(lithos_ids)[:5]}")
        if render_ids:
            logger.info(f"Sample render target IDs: {list(render_ids)[:5]}")
        return []
    
    logger.info(f"Found {len(product_ids_with_renders)} products with both LITHOS encodings and render targets")
    
    # Check for FLUXA data on render targets
    logger.info(f"Searching for FLUXA data in: {GCS_FLUXA_DATA_PATH}")
    
    keypoints_blobs = list(bucket.list_blobs(prefix=f"{GCS_FLUXA_DATA_PATH}/keypoints"))
    segmentation_blobs = list(bucket.list_blobs(prefix=f"{GCS_FLUXA_DATA_PATH}/segmentation"))
    
    # Get render target IDs with FLUXA keypoints
    keypoints_ids = set()
    for blob in keypoints_blobs:
        if blob.name.endswith('.npy'):
            render_id = os.path.splitext(os.path.basename(blob.name))[0]
            keypoints_ids.add(render_id)
    
    # Get render target IDs with FLUXA segmentation
    segmentation_ids = set()
    for blob in segmentation_blobs:
        if blob.name.endswith('.png'):
            render_id = os.path.splitext(os.path.basename(blob.name))[0]
            segmentation_ids.add(render_id)
    
    # Find render targets with FLUXA data (to use as "person images")
    valid_person_render_ids = render_ids.intersection(keypoints_ids.union(segmentation_ids))
    
    if not valid_person_render_ids:
        logger.warning("⚠️ No render targets with FLUXA data found!")
        return []
    
    logger.info(f"Found {len(valid_person_render_ids)} render targets with FLUXA data")
    
    # Create triplets where:
    # - product_id: Product to try on
    # - person_id: Render target to use as person image (must be different from product)
    # - render_target_id: Render target showing the product being worn (same ID as product)
    triplets = []
    
    # For each product with a render target
    for product_id in product_ids_with_renders:
        # Select multiple person render targets for this product
        available_person_renders = [r for r in valid_person_render_ids if r != product_id]
        if not available_person_renders:
            continue
            
        num_persons = min(args.triplets_per_product, len(available_person_renders))
        selected_persons = random.sample(available_person_renders, num_persons)
        
        for person_id in selected_persons:
            # The render target ID is the same as the product ID
            triplets.append({
                'product_id': product_id,           # Product to try on
                'person_id': person_id,             # Different render target to use as person
                'render_target_id': product_id      # Render target showing the product
            })
    
    if not triplets:
        logger.warning("⚠️ No valid triplets could be created!")
        return []
    
    # Write CSV file
    csv_content = "product_id,person_id,render_target_id\n"
    for t in triplets:
        csv_content += f"{t['product_id']},{t['person_id']},{t['render_target_id']}\n"
    
    # Save locally
    local_csv_path = os.path.join(LOCAL_TEMP_DIR, args.triplets_csv_path)
    with open(local_csv_path, 'w') as f:
        f.write(csv_content)
    
    # Upload to GCS
    gcs_csv_path = f"{GCS_BASE_PATH}/{args.triplets_csv_path}"
    blob = bucket.blob(gcs_csv_path)
    blob.upload_from_filename(local_csv_path)
    
    logger.info(f"✅ Generated and uploaded {len(triplets)} triplets to {gcs_csv_path}")
    return triplets
def load_triplets_csv(csv_path):
    """Load the CSV file that maps between product IDs, person IDs, and render target IDs."""
    logger.info(f"📂 Loading triplets from CSV: {csv_path}")
    
    try:
        # Get the blob
        blob = bucket.blob(csv_path)
        if not blob.exists():
            logger.error(f"❌ CSV file not found: {csv_path}")
            # Try alternative paths
            alternative_paths = [
                f"{GCS_BASE_PATH}/train_triplets.csv",
                f"{GCS_BASE_PATH}/inputs/train_triplets.csv"
            ]
            for alt_path in alternative_paths:
                alt_blob = bucket.blob(alt_path)
                if alt_blob.exists():
                    logger.info(f"✅ Found CSV at alternative path: {alt_path}")
                    blob = alt_blob
                    break
            else:
                return []
        
        # Download and parse CSV
        csv_content = blob.download_as_string().decode('utf-8')
        triplets = []
        
        for line in csv_content.splitlines():
            if not line.strip() or line.startswith('product_id'):  # Skip header or empty lines
                continue
                
            try:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    product_id = parts[0].strip()
                    person_id = parts[1].strip()
                    render_target_id = parts[2].strip()
                    
                    # Add to triplets list
                    triplets.append({
                        'product_id': product_id,
                        'person_id': person_id,
                        'render_target_id': render_target_id
                    })
            except Exception as e:
                logger.warning(f"⚠️ Error parsing CSV line: {line} - {e}")
                continue
        
        logger.info(f"✅ Loaded {len(triplets)} triplets from CSV")
        return triplets
        
    except Exception as e:
        logger.error(f"❌ Error loading CSV file: {e}")
        return []

def find_matching_samples(training=True):
    """Find samples with all necessary components for training."""
    # Try to load the triplets CSV
    csv_path = f"{GCS_BASE_PATH}/{args.triplets_csv_path}"
    triplets = load_triplets_csv(csv_path)
    
    if not triplets and (args.generate_triplets or training):
        # Generate new triplets if none exist
        logger.info("📊 No existing triplets found. Generating new triplets...")
        triplets = generate_training_triplets_csv()
        
        if not triplets:
            logger.error("❌ Failed to generate triplets!")
            return []
    
    # Instead of checking each triplet individually, which is slow,
    # let's just return the triplets since we already verified files exist
    # during triplet generation
    logger.info(f"📊 Using {len(triplets)} triplets for {'training' if training else 'validation'}")
    
    # Limit the number of samples if needed
    max_count = args.max_samples if training else args.max_val_samples
    return triplets[:max_count] if max_count else triplets

def download_image_from_gcs(image_id, prefix_path):
    """Download and preprocess an image from GCS."""
    logger.info(f"[DOWNLOAD] Attempting to download image: {image_id} from {prefix_path}")
    
    for ext in ['.jpg', '.png', '.jpeg']:
        blob_path = f"{prefix_path}/{image_id}{ext}"
        logger.info(f"[DOWNLOAD] Checking {blob_path}")
        blob = bucket.blob(blob_path)
        
        if blob.exists():
            try:
                logger.info(f"[DOWNLOAD] Found blob, downloading {blob_path}")
                image_bytes = blob.download_as_bytes()
                logger.info(f"[DOWNLOAD] Downloaded {len(image_bytes)} bytes")
                img = tf.image.decode_image(image_bytes, channels=3)
                logger.info(f"[DOWNLOAD] Decoded image: {img.shape}")
                img = tf.image.resize(img, (INPUT_HEIGHT, INPUT_WIDTH))
                logger.info(f"[DOWNLOAD] Resized image to {img.shape}")
                img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
                logger.info(f"[DOWNLOAD] Successfully processed image: {image_id}")
                return img
            except Exception as e:
                logger.warning(f"⚠️ [DOWNLOAD] Error loading image {image_id}: {e}")
                import traceback
                logger.warning(traceback.format_exc())
    
    # If no image found or loading failed, return a blank image
    logger.warning(f"⚠️ [DOWNLOAD] No image found for {image_id} in {prefix_path}, returning blank image")
    return tf.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=tf.float32)

def download_render_target_from_gcs(render_target_id, training=True):
    """Download and preprocess a render target image from GCS."""
    prefix_path = GCS_RENDER_TARGETS_PATH if training else GCS_VAL_RENDER_TARGETS_PATH
    logger.info(f"[RENDER] Attempting to download render target: {render_target_id} from {prefix_path}")
    
    for ext in ['.jpg', '.png', '.jpeg']:
        blob_path = f"{prefix_path}/{render_target_id}{ext}"
        logger.info(f"[RENDER] Checking {blob_path}")
        blob = bucket.blob(blob_path)
        
        if blob.exists():
            try:
                logger.info(f"[RENDER] Found blob, downloading {blob_path}")
                image_bytes = blob.download_as_bytes()
                logger.info(f"[RENDER] Downloaded {len(image_bytes)} bytes")
                img = tf.image.decode_image(image_bytes, channels=3)
                logger.info(f"[RENDER] Decoded image: {img.shape}")
                img = tf.image.resize(img, (INPUT_HEIGHT, INPUT_WIDTH))
                logger.info(f"[RENDER] Resized image to {img.shape}")
                img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
                logger.info(f"[RENDER] Successfully processed render target: {render_target_id}")
                return img
            except Exception as e:
                logger.warning(f"⚠️ [RENDER] Error loading render target {render_target_id}: {e}")
                import traceback
                logger.warning(traceback.format_exc())
    
    # If no image found or loading failed, return a blank image
    logger.warning(f"⚠️ [RENDER] No render target found for {render_target_id}, returning blank image")
    return tf.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=tf.float32)
def load_fluxa_data(person_id, training=True):
    """Load FLUXA data components with robust error handling."""
    prefix = GCS_FLUXA_DATA_PATH if training else GCS_VAL_FLUXA_DATA_PATH
    
    # Default empty data
    keypoints_heatmap = tf.zeros((KEYPOINT_H, KEYPOINT_W, NUM_KEYPOINTS), dtype=tf.float32)
    segmentation = tf.zeros((INPUT_HEIGHT, INPUT_WIDTH, 1), dtype=tf.float32)
    surface_normals = tf.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=tf.float32)
    lighting = tf.zeros(9, dtype=tf.float32)  # 9 spherical harmonic coefficients
    
    try:
        # Load keypoints
        keypoints_path = f"{prefix}/keypoints/{person_id}.npy"
        keypoints_blob = bucket.blob(keypoints_path)
        if keypoints_blob.exists():
            try:
                with io.BytesIO(keypoints_blob.download_as_bytes()) as f:
                    keypoints_data = np.load(f, allow_pickle=True)
                    
                    # Handle heatmap-format keypoints (likely in shape H×W×NUM_KEYPOINTS)
                    if len(keypoints_data.shape) == 3 and keypoints_data.shape[2] == NUM_KEYPOINTS:
                        logger.info(f"Using heatmap keypoints with shape {keypoints_data.shape}")
                        
                        # Resize to expected KEYPOINT_H × KEYPOINT_W dimensions if needed
                        if keypoints_data.shape[:2] != (KEYPOINT_H, KEYPOINT_W):
                            keypoints_data_resized = np.zeros((KEYPOINT_H, KEYPOINT_W, NUM_KEYPOINTS), dtype=np.float32)
                            
                            for i in range(NUM_KEYPOINTS):
                                keypoints_data_resized[:, :, i] = cv2.resize(
                                    keypoints_data[:, :, i],
                                    (KEYPOINT_W, KEYPOINT_H),
                                    interpolation=cv2.INTER_LINEAR
                                )
                            keypoints_data = keypoints_data_resized
                        
                        # Ensure values are in a reasonable range
                        if np.max(keypoints_data) > 1.0:
                            keypoints_data = keypoints_data / np.max(keypoints_data)
                            
                        keypoints_heatmap = tf.convert_to_tensor(keypoints_data, dtype=tf.float32)
                    
                    # Handle coordinate-format keypoints (NUM_KEYPOINTS, 2)
                    elif len(keypoints_data.shape) == 2 and keypoints_data.shape[1] == 2:
                        logger.info(f"Converting coordinate keypoints to heatmaps: {keypoints_data.shape}")
                        keypoints_heatmap = tf.zeros((KEYPOINT_H, KEYPOINT_W, NUM_KEYPOINTS), dtype=tf.float32)
                        
                        # Create a heatmap representation
                        for i in range(min(NUM_KEYPOINTS, keypoints_data.shape[0])):
                            # Get x,y coordinates
                            x_norm, y_norm = keypoints_data[i]
                            
                            # Scale to heatmap dimensions
                            x_scaled = int(np.clip(x_norm * KEYPOINT_W, 0, KEYPOINT_W - 1))
                            y_scaled = int(np.clip(y_norm * KEYPOINT_H, 0, KEYPOINT_H - 1))
                            
                            # Create a small Gaussian centered at this keypoint
                            heatmap = np.zeros((KEYPOINT_H, KEYPOINT_W), dtype=np.float32)
                            
                            for y in range(KEYPOINT_H):
                                for x in range(KEYPOINT_W):
                                    dist_squared = (x - x_scaled)**2 + (y - y_scaled)**2
                                    # Gaussian falloff
                                    if dist_squared < 25:  # 5×5 area
                                        heatmap[y, x] = np.exp(-dist_squared / 4.0) * 0.5
                            
                            keypoints_heatmap = tf.tensor_scatter_nd_update(
                                keypoints_heatmap,
                                [[i] for i in range(KEYPOINT_H * KEYPOINT_W)],
                                tf.reshape(tf.convert_to_tensor(heatmap, dtype=tf.float32), [-1])
                            )
                    else:
                        logger.warning(f"⚠️ Unexpected keypoints shape: {keypoints_data.shape}, using zeros")
            except Exception as e:
                logger.warning(f"⚠️ Error loading keypoints for {person_id}: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # Load segmentation
        segmentation_path = f"{prefix}/segmentation/{person_id}.png"
        seg_blob = bucket.blob(segmentation_path)
        if seg_blob.exists():
            try:
                with io.BytesIO(seg_blob.download_as_bytes()) as f:
                    segmentation_img = Image.open(f)
                    seg_array = np.array(segmentation_img)
                    
                    # Convert to proper format
                    if len(seg_array.shape) == 2:  # Single channel
                        seg_array = seg_array / 255.0  # Normalize
                        seg_array = np.expand_dims(seg_array, axis=-1)  # Add channel dimension
                    elif len(seg_array.shape) == 3 and seg_array.shape[2] == 4:  # RGBA
                        seg_array = seg_array[..., 3:4] / 255.0  # Use alpha channel
                    elif len(seg_array.shape) == 3 and seg_array.shape[2] == 3:  # RGB
                        # Convert RGB to grayscale
                        seg_array = np.mean(seg_array, axis=2, keepdims=True) / 255.0
                    
                    # Resize if needed
                    if seg_array.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
                        seg_array = cv2.resize(
                            seg_array, 
                            (INPUT_WIDTH, INPUT_HEIGHT), 
                            interpolation=cv2.INTER_NEAREST
                        )
                        if len(seg_array.shape) == 2:
                            seg_array = np.expand_dims(seg_array, axis=-1)
                    
                    segmentation = tf.convert_to_tensor(seg_array, dtype=tf.float32)
            except Exception as e:
                logger.warning(f"⚠️ Error loading segmentation for {person_id}: {e}")
        
        # Load lighting
        lighting_path = f"{prefix}/lighting/{person_id}.npy"
        lighting_blob = bucket.blob(lighting_path)
        if lighting_blob.exists():
            try:
                with io.BytesIO(lighting_blob.download_as_bytes()) as f:
                    lighting_data = np.load(f, allow_pickle=True)
                    # Ensure shape is correct
                    if len(lighting_data.shape) == 1:
                        # Pad or truncate to exactly 9 values
                        if lighting_data.shape[0] > 9:
                            lighting_data = lighting_data[:9]
                        else:
                            lighting_data = np.pad(lighting_data, (0, max(0, 9 - lighting_data.shape[0])))
                        lighting = tf.convert_to_tensor(lighting_data, dtype=tf.float32)
            except Exception as e:
                logger.warning(f"⚠️ Error loading lighting for {person_id}: {e}")
        
        # Load surface normals
        normals_path = f"{prefix}/surface_normals/{person_id}.npy"
        normals_blob = bucket.blob(normals_path)
        if normals_blob.exists():
            try:
                with io.BytesIO(normals_blob.download_as_bytes()) as f:
                    normals_data = np.load(f, allow_pickle=True)
                    
                    # Ensure proper dimensions
                    if len(normals_data.shape) == 2 and normals_data.shape[1] == 3:
                        # If it's a flattened array of 3D vectors, reshape to image
                        height = width = int(np.sqrt(normals_data.shape[0]))
                        normals_data = normals_data.reshape(height, width, 3)
                    
                    # Resize if needed
                    if normals_data.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
                        normals_data = cv2.resize(
                            normals_data, 
                            (INPUT_WIDTH, INPUT_HEIGHT), 
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    # Normalize if not already in [-1, 1] range
                    if np.max(normals_data) > 1.0 or np.min(normals_data) < -1.0:
                        max_val = np.max(np.abs(normals_data))
                        normals_data = normals_data / max_val
                    
                    surface_normals = tf.convert_to_tensor(normals_data, dtype=tf.float32)
            except Exception as e:
                logger.warning(f"⚠️ Error loading surface normals for {person_id}: {e}")
    
    except Exception as e:
        logger.warning(f"⚠️ Error loading FLUXA data for {person_id}: {e}")
    
    # Final tensor existence check
    if keypoints_heatmap is None or segmentation is None or surface_normals is None or lighting is None:
        logger.warning(f"⚠️ Some FLUXA tensors are None for {person_id}, using default values")
        if keypoints_heatmap is None:
            keypoints_heatmap = tf.zeros((KEYPOINT_H, KEYPOINT_W, NUM_KEYPOINTS), dtype=tf.float32)
        if segmentation is None:
            segmentation = tf.zeros((INPUT_HEIGHT, INPUT_WIDTH, 1), dtype=tf.float32)
        if surface_normals is None:
            surface_normals = tf.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=tf.float32)
        if lighting is None:
            lighting = tf.zeros(9, dtype=tf.float32)
    
    return {
        'keypoints': keypoints_heatmap,
        'segmentation': segmentation,
        'surface_normals': surface_normals,
        'lighting': lighting
    }

def load_lithos_encoding(image_id, training=True):
    """Load LITHOS encoding with support for both old and new format NPZ files."""
    prefix = GCS_LITHOS_ENCODING_PATH if training else GCS_VAL_LITHOS_ENCODING_PATH
    blob_path = f"{prefix}/{image_id}.npz"
    logger.info(f"[LITHOS] Loading LITHOS encoding: {blob_path}")
    blob = bucket.blob(blob_path)
    
    # Default encoding with appropriate shape
    encoding = tf.zeros((LITHOS_HEIGHT, LITHOS_WIDTH, LITHOS_CHANNELS), dtype=tf.float32)
    material_properties = tf.zeros(4, dtype=tf.float32)  # [reflectivity, roughness, metalness, transparency]
    material_class = tf.zeros(16, dtype=tf.float32)  # 16 material classes one-hot encoding
    
    if blob.exists():
        logger.info(f"[LITHOS] Found LITHOS blob, loading: {blob_path}")
        try:
            # Download and load the NPZ file
            with io.BytesIO(blob.download_as_bytes()) as f:
                logger.info(f"[LITHOS] Downloaded NPZ file for {image_id}")
                data = np.load(f, allow_pickle=True)
                logger.info(f"[LITHOS] NPZ loaded, keys: {list(data.keys())}")
                
                # Process encoding - support both old format ('base_encoding') and new format ('encoding')
                if 'encoding' in data:
                    # New direct LITHOS format
                    logger.info(f"[LITHOS] Processing 'encoding' key (direct LITHOS)")
                    enc = data['encoding']
                elif 'base_encoding' in data:
                    # Old neural LITHOS format
                    logger.info(f"[LITHOS] Processing 'base_encoding' key (neural LITHOS)")
                    enc = data['base_encoding']
                else:
                    # Try to use the first array as a fallback
                    logger.warning(f"[LITHOS] No recognized encoding key found, using first array")
                    first_key = list(data.keys())[0]
                    enc = data[first_key]
                
                logger.info(f"[LITHOS] Base encoding shape: {enc.shape}")
                
                # Check for NaN values
                if np.isnan(enc).any():
                    logger.warning(f"⚠️ [LITHOS] NaN values found in LITHOS encoding for {image_id}, replacing with zeros")
                    enc = np.nan_to_num(enc)
                
                # Resize if needed
                if enc.shape[:2] != (LITHOS_HEIGHT, LITHOS_WIDTH):
                    logger.info(f"[LITHOS] Resizing from {enc.shape[:2]} to {LITHOS_HEIGHT}x{LITHOS_WIDTH}")
                    enc = cv2.resize(
                        enc, 
                        (LITHOS_WIDTH, LITHOS_HEIGHT), 
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Handle channel differences
                if enc.shape[-1] != LITHOS_CHANNELS:
                    logger.info(f"[LITHOS] Adjusting channels from {enc.shape[-1]} to {LITHOS_CHANNELS}")
                    if enc.shape[-1] < LITHOS_CHANNELS:
                        # Expand channels if needed
                        padding = [(0, 0), (0, 0), (0, LITHOS_CHANNELS - enc.shape[-1])]
                        enc = np.pad(enc, padding, mode='constant')
                    else:
                        # Truncate channels if too many
                        enc = enc[..., :LITHOS_CHANNELS]
                
                # Ensure values are within a reasonable range
                enc = np.clip(enc, 0.0, 1.0)
                logger.info(f"[LITHOS] Final encoding shape: {enc.shape}")
                
                encoding = tf.convert_to_tensor(enc, dtype=tf.float32)
                
                # Process material properties if available
                if 'material_properties' in data:
                    logger.info(f"[LITHOS] Loading material properties")
                    props = data['material_properties']
                    # Ensure we have 4 properties
                    if len(props) != 4:
                        props = np.pad(props[:min(len(props), 4)], (0, max(0, 4-len(props))), 'constant', constant_values=0.5)
                    material_properties = tf.convert_to_tensor(props, dtype=tf.float32)
                
                # Process material class if available
                if 'material_class' in data:
                    logger.info(f"[LITHOS] Loading material class")
                    class_data = data['material_class']
                    # Ensure we have 16 classes
                    if len(class_data) != 16:
                        class_data = np.pad(class_data[:min(len(class_data), 16)], (0, max(0, 16-len(class_data))), 'constant')
                    material_class = tf.convert_to_tensor(class_data, dtype=tf.float32)
                
                logger.info(f"[LITHOS] Successfully loaded LITHOS encoding for {image_id}")
        
        except Exception as e:
            logger.warning(f"⚠️ [LITHOS] Error loading LITHOS encoding for {image_id}: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    else:
        logger.warning(f"⚠️ [LITHOS] LITHOS blob not found: {blob_path}")
    
    return {
        'base_encoding': encoding,  # Still return as 'base_encoding' for compatibility with rest of code
        'material_properties': material_properties,
        'material_class': material_class
    }

def generate_clothing_mask(person_image, segmentation, keypoints_heatmap):
    """
    Create a clothing mask focusing on the torso region by using FLUXA keypoint heatmaps.
    Explicitly excludes the head/face region.
    """
    # For safety, immediately create the fallback mask
    person_mask = tf.cast(segmentation > 0.5, tf.float32)
    height = tf.shape(segmentation)[0]
    width = tf.shape(segmentation)[1]
    
    # Create a blank mask of zeros
    fallback_mask = tf.zeros_like(person_mask)
    
    # Only include the middle section (torso) - more aggressive exclusion of head
    torso_start = height * 25 // 100  # Start lower to ensure head is excluded (was 15%)
    torso_end = height * 60 // 100    # Include more of the torso (was 40%)
    
    # Set only the torso region to ones
    torso_region = tf.ones((torso_end - torso_start, width, 1), dtype=tf.float32)
    fallback_mask = tf.tensor_scatter_nd_update(
        fallback_mask,
        [[i, j, 0] for i in range(torso_start, torso_end) for j in range(width)],
        tf.reshape(torso_region, [-1])
    )
    
    # Multiply with person segmentation to only include the person's body
    fallback_mask = fallback_mask * person_mask
    
    try:
        # Check if keypoints are available and valid
        if keypoints_heatmap is None or tf.reduce_sum(keypoints_heatmap) < 1e-6:
            return fallback_mask
        
        # These indices correspond to upper body keypoints in standard COCO format:
        # 0: nose, 1: left eye, 2: right eye, 3: left ear, 4: right ear, 
        # 5: left shoulder, 6: right shoulder, 7: left elbow, 8: right elbow,
        # 9: left wrist, 10: right wrist, 11: left hip, 12: right hip
        
        # Face keypoints to determine the head region to EXCLUDE
        FACE_KEYPOINTS = [0, 1, 2, 3, 4]  # nose, eyes, ears
        
        # Torso keypoints to determine the region to INCLUDE
        TORSO_KEYPOINTS = [5, 6, 7, 8, 9, 10, 11, 12]  # shoulders, elbows, wrists, hips
        
        # Convert keypoints to the right size if needed
        if keypoints_heatmap.shape[0] != segmentation.shape[0] or keypoints_heatmap.shape[1] != segmentation.shape[1]:
            keypoints_resized = tf.image.resize(
                keypoints_heatmap, 
                [tf.shape(segmentation)[0], tf.shape(segmentation)[1]]
            )
        else:
            keypoints_resized = keypoints_heatmap
        
        # Extract torso keypoints
        torso_heatmap = tf.zeros_like(segmentation)
        for idx in TORSO_KEYPOINTS:
            if idx < keypoints_resized.shape[-1]:
                keypoint_slice = keypoints_resized[..., idx:idx+1]
                torso_heatmap += keypoint_slice
        
        # Extract face keypoints
        face_heatmap = tf.zeros_like(segmentation)
        for idx in FACE_KEYPOINTS:
            if idx < keypoints_resized.shape[-1]:
                keypoint_slice = keypoints_resized[..., idx:idx+1]
                face_heatmap += keypoint_slice
        
        # Process only if both face and torso keypoints are detected
        if tf.reduce_sum(torso_heatmap) > 0.1 and tf.reduce_sum(face_heatmap) > 0.1:
            # Find vertical bounds of keypoints
            y_torso_activations = tf.reduce_max(torso_heatmap, axis=[1, 2])
            torso_y_indices = tf.where(y_torso_activations > 0)
            
            y_face_activations = tf.reduce_max(face_heatmap, axis=[1, 2])
            face_y_indices = tf.where(y_face_activations > 0)
            
            if tf.size(torso_y_indices) > 0 and tf.size(face_y_indices) > 0:
                # Get min and max y positions of torso keypoints
                min_torso_y = tf.reduce_min(tf.cast(torso_y_indices, tf.float32))
                max_torso_y = tf.reduce_max(tf.cast(torso_y_indices, tf.float32))
                
                # Get max y position of face
                max_face_y = tf.reduce_max(tf.cast(face_y_indices, tf.float32))
                
                # Add larger margin below face to ensure we don't mask part of the face
                face_bottom_y = max_face_y + 15  # Increased margin (was 5)
                
                # Create mask from below face to below torso
                height_float = tf.cast(height, tf.float32)
                
                # Define torso range: from below face to below hips
                torso_min_y = tf.cast(tf.maximum(0.0, face_bottom_y), tf.int32)
                torso_max_y = tf.cast(tf.minimum(height_float, max_torso_y + height_float * 0.15), tf.int32)
                
                # Create mask that's 1 in torso region, 0 elsewhere
                y_indices = tf.range(0, height, dtype=tf.int32)
                y_indices = tf.reshape(y_indices, [-1, 1, 1])
                
                y_mask = tf.cast(
                    tf.logical_and(
                        y_indices >= torso_min_y,
                        y_indices <= torso_max_y
                    ), 
                    tf.float32
                )
                
                # Apply vertical mask to person segmentation
                torso_mask = person_mask * y_mask
                
                # If torso mask has adequate coverage, use it
                if tf.reduce_sum(torso_mask) > 100:
                    return torso_mask
        
        # If we get here, fall back to the basic approach
        return fallback_mask
        
    except Exception as e:
        # If anything goes wrong, just use the fallback top-half mask
        print(f"Error in generate_clothing_mask: {e}, using fallback")
        return fallback_mask

def extract_clean_clothing(lithos_encoding):
    """
    Extract only the clothing from LITHOS encoding by removing background.
    Uses simple thresholding and connected component analysis.
    
    Args:
        lithos_encoding: The LITHOS encoding of the clothing (HxWxC)
        
    Returns:
        Clean clothing mask and cleaned encoding with background removed
    """
    # Convert to numpy if needed
    if isinstance(lithos_encoding, tf.Tensor):
        lithos_encoding = lithos_encoding.numpy()
    
    # Create a grayscale version for segmentation
    if lithos_encoding.shape[-1] >= 3:
        # Use RGB channels
        grayscale = np.mean(lithos_encoding[..., :3], axis=-1)
    else:
        # Use whatever channels are available
        grayscale = np.mean(lithos_encoding, axis=-1)
    
    # Create a binary mask using Otsu's thresholding
    # This works well for clothing on white/light backgrounds
    if np.max(grayscale) > 0:
        # Normalize to 0-255 range for cv2 functions
        gray_norm = (grayscale / np.max(grayscale) * 255).astype(np.uint8)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_norm, (5, 5), 0)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Find the largest connected component (likely the clothing)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
        
        # If components were found
        if num_labels > 1:
            # Skip the first component (usually background)
            largest_label = 1
            largest_area = stats[1, cv2.CC_STAT_AREA]
            
            # Find the largest non-background component
            for i in range(2, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > largest_area:
                    largest_label = i
                    largest_area = stats[i, cv2.CC_STAT_AREA]
            
            # Create the final mask
            clothing_mask = (labels == largest_label).astype(np.float32)
            
            # Dilate slightly to ensure coverage
            kernel = np.ones((3, 3), np.uint8)
            clothing_mask = cv2.dilate(clothing_mask, kernel, iterations=1)
        else:
            # If no components found, use a simple threshold
            clothing_mask = (grayscale < 0.9).astype(np.float32)
    else:
        # Fallback for empty encodings
        clothing_mask = np.zeros_like(grayscale, dtype=np.float32)
    
    # Create the clean encoding with background removed
    clean_encoding = np.copy(lithos_encoding)
    for c in range(lithos_encoding.shape[-1]):
        clean_encoding[..., c] = lithos_encoding[..., c] * clothing_mask
    
    # Expand dimensions of mask if needed
    if len(clothing_mask.shape) == 2:
        clothing_mask = np.expand_dims(clothing_mask, axis=-1)
    
    return clothing_mask, clean_encoding

def create_adaptive_clothing_mask(segmentation, aligned_clothing, keypoint_coords, 
                                 y_offset, x_offset, height, width):
    """
    Create a more accurate clothing mask that fully captures the garment shape.
    Includes robust error handling and safer operations with wrist connections.
    """
    # Initialize mask with zeros
    target_height, target_width = segmentation.shape[:2]
    mask = np.zeros((target_height, target_width), dtype=np.float32)
    
    # Get segmentation as single channel with proper error handling
    try:
        seg_mask = np.squeeze(segmentation)
        if seg_mask.max() > 1.0:  # Normalize if needed
            seg_mask = seg_mask / 255.0
    except Exception as e:
        print(f"Error processing segmentation: {e}")
        seg_mask = np.ones((target_height, target_width), dtype=np.float32) * 0.5
    
    # Extract clothing intensity to identify the actual clothing shape
    try:
        clothing_intensity = np.mean(aligned_clothing, axis=2)
        # Create a binary mask of where the clothing exists (non-zero pixels)
        clothing_exists = (clothing_intensity > 0.05).astype(np.float32)
    except Exception as e:
        print(f"Error extracting clothing intensity: {e}")
        clothing_exists = np.zeros((target_height, target_width), dtype=np.float32)
    
    # Use segmentation to constrain within the body, but with wider margins
    try:
        dilated_seg = cv2.dilate(
            seg_mask.astype(np.float32), 
            np.ones((15, 15), np.uint8),
            iterations=2
        )
    except Exception as e:
        print(f"Error dilating segmentation: {e}")
        dilated_seg = seg_mask.copy()
    
    # Combine clothing shape with expanded segmentation
    combined_mask = clothing_exists * dilated_seg
    
    # Add sleeve extensions if keypoints are available
    if keypoint_coords and isinstance(keypoint_coords, dict):
        sleeve_mask = np.zeros_like(mask)
        
        # Process shoulders and arms with safety checks
        for side in ['left', 'right']:
            shoulder_key = f'{side}_shoulder'
            elbow_key = f'{side}_elbow'
            wrist_key = f'{side}_wrist'
            
            # Only proceed if we have shoulder keypoints
            if shoulder_key in keypoint_coords:
                try:
                    # Get shoulder coordinates
                    shoulder_data = keypoint_coords[shoulder_key]
                    if len(shoulder_data) >= 2:
                        shoulder_y, shoulder_x = shoulder_data[0], shoulder_data[1]
                        shoulder_y_int = int(min(max(0, shoulder_y), target_height-1))
                        shoulder_x_int = int(min(max(0, shoulder_x), target_width-1))
                        shoulder_point = (shoulder_x_int, shoulder_y_int)
                        
                        # Check for elbow
                        elbow_point = None
                        if elbow_key in keypoint_coords:
                            elbow_data = keypoint_coords[elbow_key]
                            if len(elbow_data) >= 2:
                                elbow_y, elbow_x = elbow_data[0], elbow_data[1]
                                elbow_y_int = int(min(max(0, elbow_y), target_height-1))
                                elbow_x_int = int(min(max(0, elbow_x), target_width-1))
                                elbow_point = (elbow_x_int, elbow_y_int)
                                
                                # Draw arm from shoulder to elbow
                                cv2.line(sleeve_mask, shoulder_point, elbow_point, 1.0, thickness=20)
                                
                                # Connect to wrist if available
                                if wrist_key in keypoint_coords:
                                    wrist_data = keypoint_coords[wrist_key]
                                    if len(wrist_data) >= 2:
                                        wrist_y, wrist_x = wrist_data[0], wrist_data[1]
                                        wrist_y_int = int(min(max(0, wrist_y), target_height-1))
                                        wrist_x_int = int(min(max(0, wrist_x), target_width-1))
                                        wrist_point = (wrist_x_int, wrist_y_int)
                                        
                                        # Draw arm from elbow to wrist
                                        cv2.line(sleeve_mask, elbow_point, wrist_point, 1.0, thickness=15)
                        else:
                            # Estimate arm position if no elbow
                            direction_x = -1 if side == 'left' else 1
                            arm_length = target_height * 0.15
                            
                            # Create estimated elbow point
                            est_elbow_x = int(min(max(0, shoulder_x + direction_x * arm_length), target_width-1))
                            est_elbow_y = int(min(max(0, shoulder_y + arm_length * 0.5), target_height-1))
                            elbow_point = (est_elbow_x, est_elbow_y)
                            
                            # Draw estimated arm
                            cv2.line(sleeve_mask, shoulder_point, elbow_point, 1.0, thickness=20)
                            
                            # If we have wrist but no elbow, connect shoulder directly to wrist
                            if wrist_key in keypoint_coords:
                                wrist_data = keypoint_coords[wrist_key]
                                if len(wrist_data) >= 2:
                                    wrist_y, wrist_x = wrist_data[0], wrist_data[1]
                                    wrist_y_int = int(min(max(0, wrist_y), target_height-1))
                                    wrist_x_int = int(min(max(0, wrist_x), target_width-1))
                                    wrist_point = (wrist_x_int, wrist_y_int)
                                    
                                    # Draw direct line from shoulder to wrist
                                    cv2.line(sleeve_mask, shoulder_point, wrist_point, 1.0, thickness=15)
                
                except Exception as e:
                    print(f"Error processing {side} arm: {e}")
                    continue
        
        # Combine sleeve mask with the main mask
        combined_mask = np.maximum(combined_mask, sleeve_mask * dilated_seg)
    
    # Final processing to smooth edges
    try:
        # Use a smaller kernel for sharper edges
        kernel_size = 5
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        final_mask = cv2.GaussianBlur(combined_mask, (kernel_size, kernel_size), 0)
    except Exception as e:
        print(f"Error smoothing mask: {e}")
        final_mask = combined_mask
    
    # Ensure the output is in float32 format
    return final_mask.astype(np.float32)

def keypoint_coords_from_heatmap(keypoints_heatmap, target_height, target_width):
    """Convert heatmap representation to coordinate dictionary with confidence values."""
    keypoint_coords = {}
    
    # Standard COCO format keypoint indices
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Convert to numpy if needed
    if isinstance(keypoints_heatmap, tf.Tensor):
        keypoints_heatmap = keypoints_heatmap.numpy()
    
    # Extract coordinates from heatmaps
    for idx, name in enumerate(keypoint_names):
        if idx < keypoints_heatmap.shape[2]:
            heat_map = keypoints_heatmap[:, :, idx]
            if np.max(heat_map) > 0.1:  # Confidence threshold
                y, x = np.unravel_index(np.argmax(heat_map), heat_map.shape)
                confidence = np.max(heat_map)
                
                # Scale to target dimensions
                x_scaled = x * target_width / keypoints_heatmap.shape[1]
                y_scaled = y * target_height / keypoints_heatmap.shape[0]
                
                keypoint_coords[name] = (y_scaled, x_scaled, confidence)
    
    return keypoint_coords

def apply_shading_from_normals(clothing_pixels, surface_normals, lighting):
    """Apply realistic shading to clothing based on surface normals and lighting."""
    # Convert inputs to numpy
    if isinstance(clothing_pixels, tf.Tensor):
        clothing_pixels = clothing_pixels.numpy()
    if isinstance(surface_normals, tf.Tensor):
        surface_normals = surface_normals.numpy()
    if isinstance(lighting, tf.Tensor):
        lighting = lighting.numpy()
    
    # Process surface normals to ensure proper shape
    if surface_normals.shape[0:2] != clothing_pixels.shape[0:2]:
        surface_normals = cv2.resize(
            surface_normals, 
            (clothing_pixels.shape[1], clothing_pixels.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    # Normalize surface normals if needed
    normals = surface_normals.copy()
    if np.max(np.abs(normals)) > 0:
        norms = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
        norms = np.where(norms > 0, norms, 1.0)
        normals = normals / norms
    
    # Simple Lambertian shading from main light direction
    # (Use first 3 components of lighting as main directional light)
    light_dir = np.array([0.0, 0.0, 1.0])  # Default: light from front
    if len(lighting) >= 3:
        light_dir = lighting[:3]
        if np.sum(light_dir**2) > 0:
            light_dir = light_dir / np.sqrt(np.sum(light_dir**2))
    
    # Calculate diffuse lighting factor
    diffuse = np.maximum(0, np.sum(normals * light_dir.reshape(1, 1, 3), axis=2))
    
    # Add ambient lighting
    ambient = 0.4
    total_lighting = ambient + (1.0 - ambient) * diffuse
    
    # Expand dims for broadcasting
    total_lighting = np.expand_dims(total_lighting, axis=2)
    
    # Apply lighting to clothing
    shaded_clothing = clothing_pixels * total_lighting
    
    # Add subtle highlights based on material properties
    specular_mask = (normals[:, :, 2] > 0.7).astype(np.float32)  # Highlight front-facing areas
    specular_mask = np.expand_dims(specular_mask, axis=2)
    
    # Apply specular highlights
    specular_strength = 0.2
    shaded_clothing = shaded_clothing + specular_mask * specular_strength
    
    # Ensure values are in valid range
    shaded_clothing = np.clip(shaded_clothing, 0, 1)
    
    return tf.convert_to_tensor(shaded_clothing, dtype=tf.float32)

def create_soft_edge_mask(mask, softness=3):
    """Create a soft-edged mask for natural blending."""
    # Convert to numpy
    if isinstance(mask, tf.Tensor):
        mask = mask.numpy()
    
    # Ensure mask is in range [0,1]
    mask_norm = np.clip(mask, 0, 1)
    
    # Apply Gaussian blur for soft edges
    if softness > 0:
        # If 3D with channels, process each channel
        if len(mask_norm.shape) == 3:
            soft_mask = np.zeros_like(mask_norm)
            for c in range(mask_norm.shape[2]):
                soft_mask[:, :, c] = cv2.GaussianBlur(
                    mask_norm[:, :, c], 
                    (softness*2+1, softness*2+1),
                    0
                )
        else:
            soft_mask = cv2.GaussianBlur(
                mask_norm, 
                (softness*2+1, softness*2+1),
                0
            )
    else:
        soft_mask = mask_norm
    
    return tf.convert_to_tensor(soft_mask, dtype=tf.float32)
def align_clothing_to_keypoints(clothing_encoding, keypoints_heatmap, segmentation, target_height, target_width):
    """
    Enhanced alignment that properly scales and positions clothing on the person.
    Completely rewritten for better clothing fitting with explicit integer handling.
    """
    # Convert to numpy if needed
    if isinstance(keypoints_heatmap, tf.Tensor):
        keypoints_heatmap = keypoints_heatmap.numpy()
    if isinstance(clothing_encoding, tf.Tensor):
        clothing_encoding = clothing_encoding.numpy()
    if isinstance(segmentation, tf.Tensor):
        segmentation = segmentation.numpy()
    
    # Clean the clothing to remove background
    try:
        clothing_mask, clean_clothing = extract_clean_clothing(clothing_encoding)
    except Exception as e:
        print(f"Error in extract_clean_clothing: {e}")
        # Fall back to original encoding
        clean_clothing = clothing_encoding
        if len(clothing_encoding.shape) == 3:
            clothing_mask = np.ones((clothing_encoding.shape[0], clothing_encoding.shape[1], 1), dtype=np.float32)
        else:
            clothing_mask = np.ones((clothing_encoding.shape[0], clothing_encoding.shape[1]), dtype=np.float32)
    
    # Extract keypoints with more emphasis on upper body points
    try:
        keypoint_coords = keypoint_coords_from_heatmap(
            keypoints_heatmap, target_height, target_width
        )
    except Exception as e:
        print(f"Error extracting keypoint coordinates: {e}")
        keypoint_coords = {}
    
    # Create aligned encoding canvas
    aligned_encoding = np.zeros((target_height, target_width, clean_clothing.shape[2]), dtype=clean_clothing.dtype)
    
    # --- IMPROVED BODY ANALYSIS ---
    # Get person dimensions from segmentation
    seg_mask = np.squeeze(segmentation)
    if seg_mask.max() > 1.0:  # Normalize if needed
        seg_mask = seg_mask / 255.0
    
    try:
        # Analyze body shape using segmentation
        body_rows = np.any(seg_mask > 0.5, axis=1)
        body_cols = np.any(seg_mask > 0.5, axis=0)
        
        if np.any(body_rows) and np.any(body_cols):
            # Get person body bounds
            body_y_indices = np.where(body_rows)[0]
            body_x_indices = np.where(body_cols)[0]
            body_top = int(body_y_indices[0])  # Integer conversion
            body_bottom = int(body_y_indices[-1])  # Integer conversion
            body_left = int(body_x_indices[0])  # Integer conversion
            body_right = int(body_x_indices[-1])  # Integer conversion
            body_height = body_bottom - body_top
            body_width = body_right - body_left
            
            # --- IMPROVED KEYPOINT ANALYSIS ---
            # Default shoulder and torso estimates
            shoulder_y = int(body_top + body_height * 0.15)  # Integer conversion
            shoulder_width = body_width * 0.8
            shoulder_center_x = body_left + body_width / 2
            
            # Try to get better estimates from keypoints if available
            if 'left_shoulder' in keypoint_coords and 'right_shoulder' in keypoint_coords:
                left_y, left_x, _ = keypoint_coords['left_shoulder']
                right_y, right_x, _ = keypoint_coords['right_shoulder']
                
                # Use average for more stability
                shoulder_y = int((left_y + right_y) / 2)  # Integer conversion
                shoulder_width = abs(right_x - left_x) * 1.2
                shoulder_center_x = (left_x + right_x) / 2
            
            # --- IMPROVED CLOTHING SCALING ---
            # Calculate clothing dimensions
            clothing_height, clothing_width = clean_clothing.shape[:2]
            
            # Scale clothing to match body proportions
            # For shirts/tops - scale based on shoulder width and upper body height
            if clothing_width > 0:
                # Target width should match shoulder width plus extra for sleeves
                target_scale = shoulder_width / clothing_width
                
                # For tops, we want them to cover from neck to waist
                waist_y = int(shoulder_y + body_height * 0.3)  # Integer conversion
                torso_height = waist_y - shoulder_y
                
                # Adjust height based on clothing aspect ratio
                clothing_aspect_ratio = clothing_height / clothing_width
                target_height_calc = target_scale * clothing_width * clothing_aspect_ratio
                
                # If clothing would be too short after scaling by width,
                # ensure it's at least as tall as the torso region
                if target_height_calc < torso_height:
                    # Rescale by height instead
                    target_scale = torso_height / clothing_height
                
                # Calculate new dimensions while preserving aspect ratio
                new_width = int(clothing_width * target_scale)  # Integer conversion
                new_height = int(clothing_height * target_scale)  # Integer conversion
                
                # --- IMPROVED CLOTHING PLACEMENT ---
                # Resize clothing
                if new_width > 0 and new_height > 0:
                    try:
                        # Resize with proper interpolation for clothing
                        resized_clothing = cv2.resize(
                            clean_clothing,
                            (new_width, new_height),
                            interpolation=cv2.INTER_LINEAR  # Changed from LANCZOS4 for compatibility
                        )
                        
                        # Calculate placement
                        # Center horizontally on shoulders
                        horizontal_offset = max(0, int(shoulder_center_x - new_width / 2))  # Integer conversion
                        
                        # Vertical placement should be adjusted based on clothing type
                        # For typical tops - place top edge slightly above shoulders
                        vertical_offset = max(0, int(shoulder_y - new_height * 0.15))  # Integer conversion
                        
                        # Ensure we stay within bounds
                        horizontal_offset = min(horizontal_offset, target_width - 1)
                        vertical_offset = min(vertical_offset, target_height - 1)
                        
                        # Calculate ending coordinates with bounds checking
                        h_end = min(horizontal_offset + new_width, target_width)
                        v_end = min(vertical_offset + new_height, target_height)
                        
                        # Calculate the valid source region size
                        src_width = h_end - horizontal_offset
                        src_height = v_end - vertical_offset
                        
                        # Place the clothing on the canvas - IMPORTANT: all indices as integers
                        if src_width > 0 and src_height > 0:
                            aligned_encoding[
                                vertical_offset:v_end,
                                horizontal_offset:h_end
                            ] = resized_clothing[:src_height, :src_width]
                    except Exception as e:
                        print(f"Error during resize/placement: {e}")
                        import traceback
                        print(traceback.format_exc())
    except Exception as e:
        print(f"Error in body analysis: {e}")
        import traceback
        print(traceback.format_exc())
    
    return aligned_encoding
def create_clothing_mask_from_keypoints(keypoints_heatmap):
    """
    Use FLUXA keypoint heatmap to create a focused clothing mask (upper torso).
    This assumes the heatmap is of shape (H, W, 17), with visible activations
    around upper body keypoints: shoulders, neck, chest.
    """
    # Keypoint indices (based on COCO format)
    UPPER_BODY_KEYPOINTS = [5, 6, 11, 12]  # left/right shoulder + left/right hip

    # Combine relevant keypoints into a single activation map
    upper_heatmaps = tf.gather(keypoints_heatmap, UPPER_BODY_KEYPOINTS, axis=-1)
    combined_heatmap = tf.reduce_max(upper_heatmaps, axis=-1)  # shape: (H, W)

    # Normalize to [0, 1]
    combined_heatmap = combined_heatmap / (tf.reduce_max(combined_heatmap) + 1e-6)

    # Threshold to get binary mask
    binary_mask = tf.cast(combined_heatmap > 0.1, tf.float32)  # adjustable threshold

    # Smooth the edges a bit
    binary_mask = tf.expand_dims(binary_mask, axis=-1)  # shape: (H, W, 1)
    binary_mask = tf.nn.avg_pool2d(binary_mask[None, ...], ksize=5, strides=1, padding='SAME')[0]
    binary_mask = tf.clip_by_value(binary_mask, 0.0, 1.0)

    return binary_mask


def create_clothing_transfer_target(person_img, lithos_data, fluxa_data, clothing_mask, render_reference):
    """
    Create target with clothing resized to fit the clothing mask.
    Focus on proper sizing and positioning.
    """
    # Start with the person image
    result = tf.identity(person_img)
    
    # Get LITHOS encoding
    lithos_pixels = lithos_data['base_encoding']
    lithos_pixels = tf.convert_to_tensor(lithos_pixels, dtype=tf.float32)
    
    # Get clothing mask dimensions
    mask = clothing_mask
    if len(mask.shape) == 2:
        mask = tf.expand_dims(mask, axis=-1)
    
    # Convert to numpy for detailed processing
    mask_np = mask.numpy() if isinstance(mask, tf.Tensor) else mask
    lithos_np = lithos_pixels.numpy() if isinstance(lithos_pixels, tf.Tensor) else lithos_pixels
    person_np = person_img.numpy() if isinstance(person_img, tf.Tensor) else person_img
    
    try:
        # Extract dimensions of the clothing mask
        mask_rows = np.any(mask_np > 0.5, axis=1)
        mask_cols = np.any(mask_np > 0.5, axis=0)
        
        if np.any(mask_rows) and np.any(mask_cols):
            mask_y_indices = np.where(mask_rows)[0]
            mask_x_indices = np.where(mask_cols)[0]
            
            # Get mask bounding box
            mask_top = mask_y_indices[0]
            mask_bottom = mask_y_indices[-1]
            mask_left = mask_x_indices[0]
            mask_right = mask_x_indices[-1]
            
            mask_height = mask_bottom - mask_top
            mask_width = mask_right - mask_left
            
            # Extract clean garment from LITHOS
            garment_mask, garment = extract_clean_clothing(lithos_np)
            
            # Calculate dimensions for garment
            garment_height, garment_width = garment.shape[:2]
            
            # CRITICAL: Scale garment to match mask dimensions precisely
            scale_w = mask_width / garment_width
            scale_h = mask_height / garment_height
            
            # Use scale factor that preserves aspect ratio but ensures coverage
            scale_factor = max(scale_w, scale_h) * 1.05  # Add 5% margin
            
            new_width = int(garment_width * scale_factor)
            new_height = int(garment_height * scale_factor)
            
            # Resize garment to match mask dimensions
            if new_width > 0 and new_height > 0:
                resized_garment = cv2.resize(garment, (new_width, new_height))
                
                # Calculate positioning to center on mask
                horizontal_offset = max(0, int(mask_left + mask_width/2 - new_width/2))
                vertical_offset = max(0, int(mask_top + mask_height/2 - new_height/2))
                
                # Place resized garment on canvas
                positioned_garment = np.zeros_like(person_np)
                
                h_end = min(horizontal_offset + new_width, positioned_garment.shape[1])
                v_end = min(vertical_offset + new_height, positioned_garment.shape[0])
                src_w = h_end - horizontal_offset
                src_h = v_end - vertical_offset
                
                if src_w > 0 and src_h > 0:
                    positioned_garment[
                        vertical_offset:v_end, 
                        horizontal_offset:h_end
                    ] = resized_garment[:src_h, :src_w]
                
                # Create mask from positioned garment
                garment_present = (np.sum(positioned_garment, axis=2, keepdims=True) > 0.01).astype(np.float32)
                
                # Convert back to tensor
                positioned_garment_tensor = tf.convert_to_tensor(positioned_garment, dtype=tf.float32)
                garment_mask_tensor = tf.convert_to_tensor(garment_present, dtype=tf.float32)
                
                # Expand mask to 3 channels if needed
                if garment_mask_tensor.shape[-1] == 1:
                    garment_mask_tensor = tf.repeat(garment_mask_tensor, 3, axis=-1)
                
                # Apply direct replacement
                result = result * (1.0 - garment_mask_tensor) + positioned_garment_tensor
                
                return result
    
    except Exception as e:
        print(f"Error in garment scaling: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Fallback to simpler approach if the above fails
    # Convert mask to 3 channels if needed
    if mask.shape[-1] == 1:
        mask = tf.repeat(mask, 3, axis=-1)
    
    # Resize LITHOS to match input dimensions
    lithos_resized = tf.image.resize(lithos_pixels, [tf.shape(person_img)[0], tf.shape(person_img)[1]])
    
    # Apply binary mask replacement
    binary_mask = tf.cast(mask > 0.1, tf.float32)
    result = tf.where(binary_mask > 0.5, lithos_resized, result)
    
    return result
def data_generator(triplets, batch_size=8, training=True):
    """Generate batches of data for PRISM training with improved clothing alignment."""
    def _generate_epoch():
        if training:
            random.shuffle(triplets)
        
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i+batch_size]
            
            person_images, product_images, render_targets = [], [], []
            fluxa_data_list, lithos_list, clothing_masks = [], [], []
            
            samples_loaded = 0
            for s in batch:
                try:
                    # Extract IDs
                    prod_id, pers_id, render_id = s['product_id'], s['person_id'], s['render_target_id']
                    
                    # Load all required data
                    p_img = download_image_from_gcs(
                        pers_id, 
                        GCS_PERSON_IMAGES_PATH if training else GCS_VAL_PERSON_IMAGES_PATH
                    )
                    
                    # We don't need to load product images for training, just use a blank image
                    c_img = tf.zeros_like(p_img)
                    
                    r_img = download_render_target_from_gcs(render_id, training)
                    
                    fluxa_data = load_fluxa_data(pers_id, training)
                    
                    lithos_data = load_lithos_encoding(prod_id, training)
                    
                    # Generate enhanced clothing mask using BOTH segmentation AND keypoints
                    mask = generate_clothing_mask(
                        p_img, 
                        fluxa_data['segmentation'], 
                        fluxa_data['keypoints']
                    )
                    
                    # Apply data augmentation if training
                    if training:
                        # Simple augmentations that won't cause numerical issues
                        if tf.random.uniform(()) > 0.5:
                            # Random horizontal flip
                            p_img = tf.image.flip_left_right(p_img)
                            c_img = tf.image.flip_left_right(c_img)
                            r_img = tf.image.flip_left_right(r_img)
                            mask = tf.image.flip_left_right(mask)
                            fluxa_data['segmentation'] = tf.image.flip_left_right(fluxa_data['segmentation'])
                            fluxa_data['surface_normals'] = tf.image.flip_left_right(fluxa_data['surface_normals'])
                            # Invert x component of normals
                            fluxa_data['surface_normals'] = tf.concat([
                                -fluxa_data['surface_normals'][..., 0:1],
                                fluxa_data['surface_normals'][..., 1:],
                            ], axis=-1)
                            
                            # Also flip keypoint heatmaps
                            fluxa_data['keypoints'] = tf.image.flip_left_right(fluxa_data['keypoints'])
                        
                        # Small brightness/contrast adjustments
                        p_img = tf.image.random_brightness(p_img, 0.1)
                        p_img = tf.image.random_contrast(p_img, 0.9, 1.1)
                        p_img = tf.clip_by_value(p_img, 0.0, 1.0)
                    
                    # Store the data
                    person_images.append(p_img)
                    product_images.append(c_img)
                    render_targets.append(r_img)
                    fluxa_data_list.append(fluxa_data)
                    lithos_list.append(lithos_data)
                    clothing_masks.append(mask)
                    
                    samples_loaded += 1
                    
                except Exception:
                    continue
            
            # Skip the batch if no valid samples
            if not person_images:
                continue
            
            # Create input dictionary
            inputs = {
                'person_image': tf.stack(person_images),
                'clothing_image': tf.stack(product_images),
                'keypoints': tf.stack([f['keypoints'] for f in fluxa_data_list]),
                'segmentation': tf.stack([f['segmentation'] for f in fluxa_data_list]),
                'surface_normals': tf.stack([f['surface_normals'] for f in fluxa_data_list]),
                'lighting': tf.stack([f['lighting'] for f in fluxa_data_list]),
                'lithos_encoding': tf.stack([l['base_encoding'] for l in lithos_list]),
                'material_properties': tf.stack([l['material_properties'] for l in lithos_list]),
                'material_class': tf.stack([l['material_class'] for l in lithos_list]),
                'clothing_mask': tf.stack(clothing_masks),
                'render_reference': tf.stack(render_targets),
            }
            
            # Create target outputs with enhanced clothing alignment
            try:
                # Process each item in the batch
                batch_outputs = []
                for p_img, l, f, m, r in zip(person_images, lithos_list, fluxa_data_list, clothing_masks, render_targets):
                    # Use the new improved alignment function
                    aligned_encoding = align_clothing_to_keypoints(
                        l['base_encoding'],
                        f['keypoints'],
                        f['segmentation'],  # Pass segmentation as well
                        INPUT_HEIGHT,
                        INPUT_WIDTH
                    )
                    
                    # Create aligned LITHOS data
                    aligned_lithos_data = {
                        'base_encoding': aligned_encoding,
                        'material_properties': l['material_properties'],
                        'material_class': l['material_class']
                    }
                    
                    # Generate improved clothes transfer target
                    output = create_clothing_transfer_target(
                        person_img=p_img,
                        lithos_data=l,  # Using aligned data
                        fluxa_data=f,
                        clothing_mask=m,
                        render_reference=r
                    )
                    batch_outputs.append(output)
                
                # Stack outputs into a batch tensor
                outputs = tf.stack(batch_outputs)
            except Exception as e:
                print(f"Error creating clothing transfer targets: {e}")
                import traceback
                print(traceback.format_exc())
                raise
            
            yield inputs, outputs
    
    # Continuous iteration for training, single pass for validation
    while True:
        try:
            yield from _generate_epoch()
        except Exception as e:
            print(f"Error in _generate_epoch: {e}")
            raise
            
        if not training:
            break

def create_prism_dataset(triplets, batch_size=8, training=True):
    """Create a TF dataset yielding (x_dict, y) where y is the render target tensor."""
    output_signature = (
        {
            'person_image':       tf.TensorSpec((None, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=tf.float32),
            'clothing_image':     tf.TensorSpec((None, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=tf.float32),
            'keypoints':          tf.TensorSpec((None, KEYPOINT_H, KEYPOINT_W, NUM_KEYPOINTS), dtype=tf.float32),
            'segmentation':       tf.TensorSpec((None, INPUT_HEIGHT, INPUT_WIDTH, 1), dtype=tf.float32),
            'surface_normals':    tf.TensorSpec((None, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=tf.float32),
            'lighting':           tf.TensorSpec((None, 9), dtype=tf.float32),
            'lithos_encoding':    tf.TensorSpec((None, LITHOS_HEIGHT, LITHOS_WIDTH, LITHOS_CHANNELS), dtype=tf.float32),
            'material_properties':tf.TensorSpec((None, 4), dtype=tf.float32),
            'material_class':     tf.TensorSpec((None, 16), dtype=tf.float32),
            'clothing_mask':      tf.TensorSpec((None, INPUT_HEIGHT, INPUT_WIDTH, 1), dtype=tf.float32),
            "render_reference": tf.TensorSpec((None, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=tf.float32),

        },
        tf.TensorSpec((None, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=tf.float32)  # This is now render_target
    )
    
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(triplets, batch_size, training),
        output_signature=output_signature
    )
    
    # Always repeat the dataset for training to ensure it doesn't run out
    if training:
        ds = ds.repeat()
    
    return ds.prefetch(tf.data.AUTOTUNE)




# =========================
# Model Building Blocks
# =========================

class ResizeLayer(Layer):
    """Custom layer for resizing with error handling."""
    def __init__(self, target_height, target_width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width
        
    def call(self, inputs):
        # Ensure inputs are float32 for numerical stability
        inputs = tf.cast(inputs, tf.float32)
        
        try:
            return tf.image.resize(
                inputs, [self.target_height, self.target_width],
                method=tf.image.ResizeMethod.BILINEAR
            )
        except tf.errors.InvalidArgumentError:
            # Fallback in case of errors
            logger.warning(f"⚠️ Error in ResizeLayer, using simple reshape")
            return tf.image.resize(
                inputs, [self.target_height, self.target_width],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
    
    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({
            'target_height': self.target_height,
            'target_width': self.target_width
        })
        return config

class UpsampleBlock(Layer):
    """Simplified upsampling block with better numerical stability."""
    
    def __init__(self, filters, kernel_size=3, use_batch_norm=True, 
                 dropout_rate=0.0, l2_reg=1e-5, name=None):
        super(UpsampleBlock, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        # Define layers
        self.upsample = UpSampling2D(size=(2, 2))
        self.conv = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_initializer='glorot_uniform',  # Xavier/Glorot for better initialization
            kernel_regularizer=regularizers.l2(l2_reg)
        )
        self.batch_norm = BatchNormalization() if use_batch_norm else None
        self.activation = LeakyReLU(0.1)  # Lower alpha for more stable gradients
        
        # Optional dropout for regularization
        self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def call(self, inputs, skip_connection=None, training=None):
        x = self.upsample(inputs)
        
        # Apply skip connection if provided
        if skip_connection is not None:
            # Ensure skip connection has the right shape
            if x.shape[1:3] != skip_connection.shape[1:3]:
                skip_connection = ResizeLayer(
                    target_height=x.shape[1],
                    target_width=x.shape[2]
                )(skip_connection)
            
            x = Concatenate()([x, skip_connection])
        
        # Apply convolution and normalization
        x = self.conv(x)
        
        if self.use_batch_norm:
            x = self.batch_norm(x, training=training)
        
        x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        
        return x

class SimpleAttentionModule(Layer):
    """A simplified attention mechanism with numerical stability considerations."""
    
    def __init__(self, channels, reduction_ratio=8, **kwargs):
        super(SimpleAttentionModule, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Reduction step
        reduced_channels = max(channels // reduction_ratio, 8)  # Ensure at least 8 channels
        
        # Layers
        self.global_avg_pool = GlobalAveragePooling2D()
        self.fc1 = Dense(reduced_channels, activation='relu', 
                         kernel_initializer='he_normal')  # He init for ReLU
        self.fc2 = Dense(channels, activation='sigmoid',
                         kernel_initializer='glorot_uniform')  # Glorot for sigmoid
    
    def call(self, inputs):
        # Channel attention mechanism
        avg_pool = self.global_avg_pool(inputs)
        avg_pool = self.fc1(avg_pool)
        avg_pool = self.fc2(avg_pool)
        
        # Reshape for broadcasting
        channel_attention = Reshape((1, 1, self.channels))(avg_pool)
        
        # Apply attention
        return inputs * channel_attention
    
    def get_config(self):
        config = super(SimpleAttentionModule, self).get_config()
        config.update({
            'channels': self.channels,
            'reduction_ratio': self.reduction_ratio
        })
        return config

def align_clothing_to_person_tensor(lithos_encoding, keypoints, segmentation, height, width):
    """TensorFlow-compatible wrapper for clothing alignment"""
    
    def _process_single_sample(args):
        encoding, kp, seg = args
        # Convert to numpy for processing
        encoding_np = encoding.numpy()
        kp_np = kp.numpy()
        seg_np = seg.numpy()
        
        # Align using the numpy function
        aligned = align_clothing_to_keypoints(
            encoding_np, kp_np, seg_np, height, width
        )
        
        # Convert back to tensor
        return tf.convert_to_tensor(aligned, dtype=tf.float32)
    
    # Process each sample in the batch
    return tf.map_fn(
        _process_single_sample,
        (lithos_encoding, keypoints, segmentation),
        fn_output_signature=tf.float32
    )
# =========================
# Simplified Model Architecture
# =========================

class SimplePRISMModel:
    """
    Simplified PRISM Model with improved numerical stability.
    This model prioritizes reliable training over complex features.
    """
    
    def __init__(self,
                input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3),
                lithos_shape=(LITHOS_HEIGHT, LITHOS_WIDTH, LITHOS_CHANNELS),
                training=False,
                learning_rate=0.0002,
                dropout_rate=0.1,
                l2_reg=1e-6,
                use_attention=False,
                clothing_weight=2.0,
                background_weight=0.5,
                perceptual_weight=0.1,
                gradient_clip=1.0):
        """
        Initialize the simplified PRISM model.
        """
        self.input_shape = input_shape
        self.lithos_shape = lithos_shape
        self.training = training
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_attention = use_attention
        self.clothing_weight = clothing_weight
        self.background_weight = background_weight
        self.perceptual_weight = perceptual_weight
        self.gradient_clip = gradient_clip
        
        # Build the model
        self._build_model()
        
        # Compile the model if in training mode
        if training:
            self._compile_model()

    def _build_model(self):
        """Build a simplified PRISM model with improved FLUXA integration."""
        
        # === Inputs ===
        person_input = Input(shape=self.input_shape, name="person_image")
        clothing_input = Input(shape=self.input_shape, name="clothing_image")
        keypoints_input = Input(shape=(KEYPOINT_H, KEYPOINT_W, NUM_KEYPOINTS), name="keypoints")
        segmentation_input = Input(shape=(self.input_shape[0], self.input_shape[1], 1), name="segmentation")
        surface_normals_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3), name="surface_normals")
        lighting_input = Input(shape=(9,), name="lighting")
        lithos_encoding_input = Input(shape=self.lithos_shape, name="lithos_encoding")
        material_properties_input = Input(shape=(4,), name="material_properties")
        material_class_input = Input(shape=(16,), name="material_class")
        clothing_mask_input = Input(shape=(self.input_shape[0], self.input_shape[1], 1), name="clothing_mask")
        render_reference_input = Input(shape=self.input_shape, name="render_reference")

        logger.info(f"Input shapes: person={self.input_shape}, lithos={self.lithos_shape}")
        
        # === Person Image Encoder ===
        x_person = Conv2D(32, 3, strides=2, padding='same',
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        kernel_initializer='he_normal')(person_input)
        x_person = BatchNormalization()(x_person)
        x_person = LeakyReLU(0.1)(x_person)
        skip1 = x_person

        x_person = Conv2D(64, 3, strides=2, padding='same',
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        kernel_initializer='he_normal')(x_person)
        x_person = BatchNormalization()(x_person)
        x_person = LeakyReLU(0.1)(x_person)
        skip2 = x_person
        
        # Log shape for debugging purposes
        logger.info(f"Person feature shape after encoding: {x_person.shape}")
        
        # === Clothing Image Encoder ===
        x_clothing = Conv2D(32, 3, strides=2, padding='same',
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        kernel_initializer='he_normal')(clothing_input)
        x_clothing = BatchNormalization()(x_clothing)
        x_clothing = LeakyReLU(0.1)(x_clothing)

        x_clothing = Conv2D(48, 3, strides=2, padding='same',
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        kernel_initializer='he_normal')(x_clothing)
        x_clothing = BatchNormalization()(x_clothing)
        x_clothing = LeakyReLU(0.1)(x_clothing)
        
        # Match dimensions for feature combination
        clothing_resized = ResizeLayer(
            target_height=int(x_person.shape[1]),
            target_width=int(x_person.shape[2]),
            name="clothing_feature_resize"
        )(x_clothing)
        logger.info(f"Clothing features shape after resize: {clothing_resized.shape}")

        # === LITHOS Encoding Preprocessing ===
        # Simple feature extraction from LITHOS encoding
        # First resize to match feature dimensions
        lithos_resized = ResizeLayer(
            target_height=int(x_person.shape[1]),
            target_width=int(x_person.shape[2]),
            name="lithos_resize"
        )(lithos_encoding_input)
        
        # Extract features from the LITHOS encoding
        lithos_features = Conv2D(64, 3, padding='same',
                                kernel_regularizer=regularizers.l2(self.l2_reg),
                                kernel_initializer='he_normal')(lithos_resized)
        lithos_features = BatchNormalization()(lithos_features)
        lithos_features = LeakyReLU(0.1)(lithos_features)
        
        lithos_features = Conv2D(96, 3, padding='same',
                                kernel_regularizer=regularizers.l2(self.l2_reg),
                                kernel_initializer='he_normal')(lithos_features)
        lithos_features = BatchNormalization()(lithos_features)
        lithos_features = LeakyReLU(0.1)(lithos_features)
        
        # === Keypoints ===
        kp = Conv2D(32, 3, padding='same',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                kernel_initializer='he_normal')(keypoints_input)
        kp = BatchNormalization()(kp)
        kp = LeakyReLU(0.1)(kp)
        
        # Resize to match person feature dimensions
        kp = ResizeLayer(
            target_height=int(x_person.shape[1]),
            target_width=int(x_person.shape[2]),
            name="keypoints_resize"
        )(kp)
        logger.info(f"Keypoints shape after resize: {kp.shape}")

        # === Segmentation ===
        seg = Conv2D(16, 3, strides=4, padding='same',
                    kernel_regularizer=regularizers.l2(self.l2_reg),
                    kernel_initializer='he_normal')(segmentation_input)
        seg = BatchNormalization()(seg)
        seg = LeakyReLU(0.1)(seg)
        
        # Resize to match person feature dimensions
        seg = ResizeLayer(
            target_height=int(x_person.shape[1]),
            target_width=int(x_person.shape[2]),
            name="segmentation_resize"
        )(seg)
        logger.info(f"Segmentation shape after resize: {seg.shape}")
        
        # === Surface Normals ===
        normals = Conv2D(24, 3, strides=4, padding='same',
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        kernel_initializer='he_normal')(surface_normals_input)
        normals = BatchNormalization()(normals)
        normals = LeakyReLU(0.1)(normals)
        
        # Resize to match person feature dimensions
        normals = ResizeLayer(
            target_height=int(x_person.shape[1]),
            target_width=int(x_person.shape[2]),
            name="normals_resize"
        )(normals)
        logger.info(f"Surface normals shape after resize: {normals.shape}")
        
        # === Lighting Parameters ===
        # Process through dense layers
        light = Dense(32, activation='relu')(lighting_input)
        light = Dense(64, activation='relu')(light)
        
        # Convert to spatial feature map for concatenation
        light = Reshape((1, 1, 64))(light)
        
        # Tile to match spatial dimensions of other features
        light_features = Lambda(
            lambda x: tf.tile(x, [1, int(x_person.shape[1]), int(x_person.shape[2]), 1])
        )(light)
        
        logger.info(f"Light features shape: {light_features.shape}")

        # === Feature Fusion ===
        # Log all shapes before concatenation to help debug
        logger.info(f"Shapes before concatenation:")
        logger.info(f"  - Person features: {x_person.shape}")
        logger.info(f"  - Clothing features: {clothing_resized.shape}")
        logger.info(f"  - Keypoints: {kp.shape}")
        logger.info(f"  - Segmentation: {seg.shape}")
        logger.info(f"  - Surface normals: {normals.shape}")
        logger.info(f"  - Light features: {light_features.shape}")
        logger.info(f"  - LITHOS features: {lithos_features.shape}")
        
        # Combine all features - ensure all have the same spatial dimensions
        features = Concatenate(name="encoder_concat")([
            x_person,          # Person features
            clothing_resized,  # Clothing features
            kp,                # Keypoint information
            seg,               # Segmentation
            normals,           # Surface normals for lighting
            light_features,    # Lighting information
            lithos_features    # Base encoding
        ])
        
        # Apply attention if requested
        if self.use_attention:
            features = SimpleAttentionModule(features.shape[-1])(features)
        
        logger.info(f"Combined features shape: {features.shape}")

        # === Decoder ===
        x = UpsampleBlock(
            filters=128, 
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg, 
            name="upsample1"
        )(features, skip2)
        
        x = UpsampleBlock(
            filters=64, 
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg, 
            name="upsample2"
        )(x, skip1)
        
        # Final upsampling/resizing if needed
        logger.info(f"Shape after upsampling: {x.shape}")
        IH, IW = self.input_shape[0], self.input_shape[1]
        h, w = int(x.shape[1]), int(x.shape[2])
        
        if h * 2 == IH and w * 2 == IW:
            logger.info("Adding final upsampling to match input dimensions")
            x = UpsampleBlock(filters=32, dropout_rate=self.dropout_rate,
                            l2_reg=self.l2_reg, name="upsample3")(x)
        elif h != IH or w != IW:
            logger.info(f"Directly resizing from {h}×{w} to {IH}×{IW}")
            x = ResizeLayer(target_height=IH, target_width=IW,
                        name="final_resize")(x)
        
        # Final output layer to generate clothing rendering
        x = Conv2D(16, 3, padding='same',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        
        x = Conv2D(3, 3, padding='same', activation='sigmoid',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                kernel_initializer='glorot_uniform',
                name="generated_clothing")(x)
        
        logger.info(f"Output shape before blending: {x.shape}")
        
        # === Simple Clothing Transfer with Basic Keras Operations ===
        # Create a simple blending using a Keras Multiply and Add layers
        
        # First create masks and inverse masks for blending
        clothing_mask_expanded = clothing_mask_input
        if clothing_mask_expanded.shape[-1] == 1:
            clothing_mask_expanded = Lambda(
                lambda m: tf.repeat(m, 3, axis=-1)
            )(clothing_mask_expanded)
        
        # Apply blur to mask for smooth edges (using a basic conv approximation is safer than tf ops)
        blurred_mask = Conv2D(3, 5, padding='same', activation='sigmoid',
                            kernel_initializer='ones', trainable=False,
                            use_bias=False)(clothing_mask_expanded)
        
        # Invert the mask for blending
        inverse_mask = Lambda(lambda m: 1.0 - m)(blurred_mask)
        
        # Apply the blending
        person_component = Multiply()([person_input, inverse_mask])
        clothing_component = Multiply()([x, blurred_mask])
        output = Add()([person_component, clothing_component])
        
        logger.info(f"Final output shape: {output.shape}")
        
        # Build model
        self.model = Model(
            inputs={
                "person_image": person_input,
                "clothing_image": clothing_input,
                "keypoints": keypoints_input,
                "segmentation": segmentation_input,
                "surface_normals": surface_normals_input,
                "lighting": lighting_input,
                "lithos_encoding": lithos_encoding_input,
                "material_properties": material_properties_input,
                "material_class": material_class_input,
                "clothing_mask": clothing_mask_input,
                "render_reference": render_reference_input
            },
            outputs=output,
            name="enhanced_prism_model"
        )
    def _compile_model(self):
        """Compile with a simpler loss function compatible with Keras."""
        
        # Define a simple loss function that doesn't try to access inputs directly
        # This works better with Keras's functional API
        def mae_loss(y_true, y_pred):
            # Simple Mean Absolute Error loss
            return tf.reduce_mean(tf.abs(y_true - y_pred))
        
        # Standard metrics
        def psnr_metric(y_true, y_pred):
            return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))
        
        def ssim_metric(y_true, y_pred):
            return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        
        # Define optimizer with gradient clipping for stability
        optimizer = Adam(
            learning_rate=self.learning_rate,
            clipnorm=self.gradient_clip
        )
        
        # Compile with standard MAE loss for now
        self.model.compile(
            optimizer=optimizer,
            loss=mae_loss,
            metrics=[
                'mae',
                psnr_metric, 
                ssim_metric
            ]
        )
    
    def save_checkpoint(self, step, epoch=0, logs=None):
        """Save model checkpoint with error handling."""
        try:
            checkpoint_name = f"prism_model_step_{step}.weights.h5"
            
            # Save locally first
            local_path = os.path.join(LOCAL_TEMP_DIR, "checkpoints", checkpoint_name)
            self.model.save_weights(local_path)
            logger.info(f"✅ Saved model checkpoint locally to {local_path}")
            
            # Upload to GCS
            gcs_path = f"{GCS_CHECKPOINT_DIR}/{checkpoint_name}"
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            logger.info(f"✅ Uploaded model checkpoint to {gcs_path}")
            
            # Save logs if provided
            if logs:
                log_name = f"prism_logs_step_{step}.json"
                local_log_path = os.path.join(LOCAL_TEMP_DIR, "logs", log_name)
                
                # Convert non-serializable values to strings
                serializable_logs = {}
                for k, v in logs.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        serializable_logs[k] = v
                    else:
                        serializable_logs[k] = str(v)
                
                with open(local_log_path, 'w') as f:
                    json.dump(serializable_logs, f)
                
                gcs_log_path = f"{GCS_LOG_DIR}/{log_name}"
                blob = bucket.blob(gcs_log_path)
                blob.upload_from_filename(local_log_path)
                logger.info(f"✅ Uploaded training logs to {gcs_log_path}")
            
            # Save metadata
            metadata = {
                "step": step,
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "input_size": f"{INPUT_HEIGHT}x{INPUT_WIDTH}",
                "lithos_shape": f"{LITHOS_HEIGHT}x{LITHOS_WIDTH}x{LITHOS_CHANNELS}",
                "learning_rate": float(self.learning_rate)
            }
            
            meta_name = f"prism_metadata_step_{step}.json"
            local_meta_path = os.path.join(LOCAL_TEMP_DIR, "logs", meta_name)
            with open(local_meta_path, 'w') as f:
                json.dump(metadata, f)
            
            gcs_meta_path = f"{GCS_LOG_DIR}/{meta_name}"
            blob = bucket.blob(gcs_meta_path)
            blob.upload_from_filename(local_meta_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving checkpoint: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_checkpoint(self, checkpoint_path=None, step=None):
        """Load model checkpoint with fallback options."""
        try:
            # Determine the checkpoint path
            if checkpoint_path:
                gcs_path = checkpoint_path
            elif step is not None:
                checkpoint_name = f"prism_model_step_{step}.weights.h5"
                gcs_path = f"{GCS_CHECKPOINT_DIR}/{checkpoint_name}"
            else:
                # Find the latest checkpoint
                blobs = list(bucket.list_blobs(prefix=f"{GCS_CHECKPOINT_DIR}/"))
                checkpoint_blobs = [b for b in blobs if b.name.endswith('.h5')]
                
                if not checkpoint_blobs:
                    logger.warning("⚠️ No checkpoints found")
                    return False
                
                # Sort by creation time (latest first)
                latest_blob = sorted(checkpoint_blobs, key=lambda b: b.updated, reverse=True)[0]
                gcs_path = latest_blob.name
                logger.info(f"✅ Found latest checkpoint: {gcs_path}")
            
            # Check if checkpoint exists
            blob = bucket.blob(gcs_path)
            if not blob.exists():
                logger.error(f"❌ Checkpoint not found: {gcs_path}")
                return False
            
            # Download to local temp directory
            local_path = os.path.join(LOCAL_TEMP_DIR, "checkpoints", os.path.basename(gcs_path))
            blob.download_to_filename(local_path)
            logger.info(f"✅ Downloaded checkpoint from {gcs_path} to {local_path}")
            
            # Load weights
            self.model.load_weights(local_path)
            logger.info(f"✅ Loaded model weights from checkpoint")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Try to find an earlier checkpoint if step was specified
            if step is not None:
                try:
                    logger.info(f"🔄 Attempting to find an earlier checkpoint...")
                    blobs = list(bucket.list_blobs(prefix=f"{GCS_CHECKPOINT_DIR}/"))
                    checkpoint_blobs = [b for b in blobs if b.name.endswith('.h5')]
                    
                    # Extract step numbers from filenames
                    step_blobs = []
                    for blob in checkpoint_blobs:
                        try:
                            blob_step = int(blob.name.split('_step_')[1].split('.h5')[0])
                            if blob_step < step:  # Only earlier steps
                                step_blobs.append((blob_step, blob))
                        except (ValueError, IndexError):
                            continue
                    
                    if step_blobs:
                        # Get the latest step that's earlier than the requested one
                        earlier_step, earlier_blob = sorted(step_blobs, key=lambda x: x[0], reverse=True)[0]
                        logger.info(f"✅ Found earlier checkpoint at step {earlier_step}")
                        
                        # Download and load
                        local_path = os.path.join(LOCAL_TEMP_DIR, "checkpoints", os.path.basename(earlier_blob.name))
                        earlier_blob.download_to_filename(local_path)
                        self.model.load_weights(local_path)
                        logger.info(f"✅ Loaded earlier checkpoint weights")
                        return True
                except Exception as fallback_e:
                    logger.error(f"❌ Fallback loading also failed: {str(fallback_e)}")
            
            return False

    def generate_visualizations(self, dataset, prefix, max_samples=4):
        """Generate and save visualization images for debugging using training data."""
        # Set up visualization directory
        vis_dir = os.path.join(LOCAL_TEMP_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            # Import matplotlib for visualization
            import matplotlib.pyplot as plt
            
            # Important: Get a batch directly from the training dataset
            # instead of creating a new dataset with validation data
            samples_seen = 0
            
            for batch in dataset.take(1):  # Just take one batch from the actual training dataset
                inputs, targets = batch
                
                # Make predictions
                predictions = self.model.predict(inputs)
                
                # Number of samples to visualize
                num_samples = min(max_samples, predictions.shape[0])
                
                for i in range(num_samples):
                    # Create a figure with subplots
                    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # Extract images for this sample
                    person_img = inputs['person_image'][i].numpy()
                    clothing_mask = inputs['clothing_mask'][i].numpy()
                    masked_overlay = np.clip(person_img + clothing_mask * np.array([0.0, 1.0, 0.0]), 0.0, 1.0)  # green overlay
                    segmentation = inputs['segmentation'][i].numpy()
                    render_target = targets[i].numpy()
                    predicted_img = predictions[i]
                    
                    # Also extract LITHOS encoding for visualization
                    lithos_enc = inputs['lithos_encoding'][i].numpy()
                    
                    # Process LITHOS encoding for visualization
                    if lithos_enc.shape[-1] >= 3:
                        # Use first 3 channels
                        lithos_vis = lithos_enc[..., :3]
                    else:
                        # If fewer than 3 channels, duplicate channels to make it viewable as RGB
                        channels_to_add = 3 - lithos_enc.shape[-1]
                        padding = [(0, 0), (0, 0), (0, channels_to_add)]
                        lithos_vis = np.pad(lithos_enc, padding, mode='constant')
                    
                    # Ensure LITHOS visualization is in proper range
                    if np.max(lithos_vis) > 1.0 or np.min(lithos_vis) < 0.0:
                        # Normalize to [0,1] range
                        min_val = np.min(lithos_vis)
                        max_val = np.max(lithos_vis)
                        if max_val > min_val:
                            lithos_vis = (lithos_vis - min_val) / (max_val - min_val)
                        else:
                            lithos_vis = np.zeros_like(lithos_vis)
                    
                    # Add debug logging
                    logger.info(f"Visualization {i+1}/{num_samples} - Person: range {np.min(person_img):.4f}-{np.max(person_img):.4f}, " 
                            f"Mask: range {np.min(clothing_mask):.4f}-{np.max(clothing_mask):.4f}, "
                            f"Target: range {np.min(render_target):.4f}-{np.max(render_target):.4f}, "
                            f"Prediction: range {np.min(predicted_img):.4f}-{np.max(predicted_img):.4f}")
                    
                    # Helper function to display an image
                    def show_image(ax, img, title):
                        # Check if image is all zeros/constant
                        if np.allclose(img, img.flat[0]):
                            logger.warning(f"⚠️ {title} is entirely constant ({img.flat[0]:.4f})")
                            
                            # Force some color to make the problem visible
                            if title == 'Person Image':
                                img = np.ones_like(img) * 0.5  # Gray
                            elif title == 'Clothing Mask':
                                img = np.ones_like(img) * 0.7  # Light gray
                            elif title == 'Segmentation':
                                img = np.ones_like(img) * 0.3  # Dark gray
                        
                        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                            # Single channel image (mask)
                            if len(img.shape) == 3:
                                img = np.squeeze(img)
                            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                        else:
                            # RGB image - ensure range is visible
                            if np.max(img) <= 1.0 and np.min(img) >= 0.0:
                                # Already in [0,1] range
                                ax.imshow(np.clip(img, 0, 1))
                            elif np.max(img) <= 255 and np.min(img) >= 0:
                                # [0,255] range
                                ax.imshow(np.clip(img, 0, 255) / 255.0)
                            else:
                                # Unknown range - normalize to [0,1]
                                img_min = np.min(img)
                                img_max = np.max(img)
                                if img_max > img_min:
                                    img = (img - img_min) / (img_max - img_min)
                                else:
                                    img = np.zeros_like(img)
                                ax.imshow(img)
                        
                        ax.set_title(title)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    
                    # Display images
                    show_image(axs[0, 0], person_img, 'Person Image')
                    show_image(axs[0, 1], clothing_mask, 'Clothing Mask')
                    show_image(axs[0, 2], segmentation, 'Segmentation')
                    
                    # Show render target and prediction
                    show_image(axs[1, 0], render_target, 'Render Target')
                    show_image(axs[1, 1], predicted_img, 'Prediction')
                    
                    # Show LITHOS encoding instead of difference
                    show_image(axs[1, 2], lithos_vis, 'LITHOS Encoding (RGB)')
                    
                    # Calculate difference metrics
                    mse = np.mean((render_target - predicted_img) ** 2)
                    mae = np.mean(np.abs(render_target - predicted_img))
                    
                    # Calculate PSNR if possible (avoiding log(0))
                    if mse > 0:
                        psnr = 10 * np.log10(1.0 / mse)  # Max value is 1.0
                    else:
                        psnr = float('inf')
                    
                    # Add overall figure title with metrics
                    plt.suptitle(f"Training Visualization - Sample {i+1}, MAE: {mae:.4f}, PSNR: {psnr:.2f}dB", fontsize=14)
                    
                    # Adjust layout and save
                    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
                    vis_path = os.path.join(vis_dir, f"{prefix}_sample_{i+1}.png")
                    plt.savefig(vis_path, dpi=150)
                    plt.close(fig)
                    
                    # Also create a separate figure for the difference visualization
                    diff_fig, diff_ax = plt.subplots(1, 1, figsize=(8, 8))
                    diff_img = np.abs(render_target - predicted_img) * 5.0  # Amplify differences
                    show_image(diff_ax, np.clip(diff_img, 0, 1), f'Difference (x5)\nMSE: {mse:.4f}, PSNR: {psnr:.2f}dB')
                    diff_path = os.path.join(vis_dir, f"{prefix}_diff_{i+1}.png")
                    plt.savefig(diff_path, dpi=150)
                    plt.close(diff_fig)
                    
                    # Upload both to GCS
                    gcs_vis_path = f"{GCS_OUTPUT_DIR}/visualizations/{prefix}_sample_{i+1}.png"
                    blob = bucket.blob(gcs_vis_path)
                    blob.upload_from_filename(vis_path)
                    
                    gcs_diff_path = f"{GCS_OUTPUT_DIR}/visualizations/{prefix}_diff_{i+1}.png"
                    blob = bucket.blob(gcs_diff_path)
                    blob.upload_from_filename(diff_path)
                    
                    samples_seen += 1
                
                logger.info(f"✅ Generated and uploaded {samples_seen} visualizations with prefix '{prefix}'")
                break  # Only process one batch
            
            # If we didn't get any samples from the dataset, log a warning
            if samples_seen == 0:
                logger.warning(f"⚠️ No samples were visualized for prefix '{prefix}' - dataset may be empty")
        
        except Exception as e:
            logger.error(f"❌ Error generating visualizations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    def train(self, train_dataset, validation_dataset=None, epochs=100, 
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            checkpoint_steps=1000, visualize=False):
        """
        Train the PRISM model with proper error handling and logging.
        
        Args:
            train_dataset: TensorFlow dataset for training
            validation_dataset: Optional TensorFlow dataset for validation
            epochs: Number of epochs to train
            initial_epoch: Initial epoch (for resuming training)
            steps_per_epoch: Number of steps per epoch
            validation_steps: Number of validation steps
            checkpoint_steps: Save checkpoint every N steps
            visualize: Whether to generate visualizations during training
        
        Returns:
            Training history
        """
        # Define callbacks
        callbacks = []
        
        # TensorBoard logging
        tensorboard_dir = os.path.join(LOCAL_TEMP_DIR, "logs", "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_callback = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        # Learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss' if validation_dataset else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss' if validation_dataset else 'loss',
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Custom callback for step-based checkpointing and visualization
        class StepCheckpointCallback(Callback):
            def __init__(self, model_obj, checkpoint_steps, visualize, val_data=None):
                super().__init__()
                self.model_obj = model_obj
                self.checkpoint_steps = checkpoint_steps
                self.visualize = visualize
                self.val_data = val_data
                self.steps = 0
                self.epoch = 0
                self.batch_losses = []
                
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch = epoch
                
            def on_epoch_end(self, epoch, logs=None):
                # Save checkpoint at the end of every epoch
                self.model_obj.save_checkpoint(f"epoch_{epoch}", epoch, logs)
                
                # Generate visualizations at the end of every epoch if enabled
                if self.visualize and self.val_data:
                    self.model_obj.generate_visualizations(self.val_data, f"epoch_{epoch}")
                
            def on_batch_begin(self, batch, logs=None):
                pass
                
            def on_batch_end(self, batch, logs=None):
                self.steps += 1
                
                # Store batch loss for averaging
                if logs and 'loss' in logs:
                    self.batch_losses.append(logs['loss'])
                
                # Save checkpoint every N steps
                if self.steps % self.checkpoint_steps == 0:
                    self.model_obj.save_checkpoint(self.steps, self.epoch, logs)
                    
                    # Generate visualizations if enabled
                    if self.visualize and self.val_data:
                        self.model_obj.generate_visualizations(self.val_data, f"step_{self.steps}")
        
        # Add step checkpoint callback
        step_callback = StepCheckpointCallback(
            model_obj=self,
            checkpoint_steps=checkpoint_steps,
            visualize=visualize,
            val_data=validation_dataset or train_dataset
        )
        callbacks.append(step_callback)
        
        # NaN and Inf checking callback
        class NanInfCallback(Callback):
            def on_batch_end(self, batch, logs=None):
                if logs and any(k in logs and (np.isnan(logs[k]) or np.isinf(logs[k])) 
                            for k in logs):
                    logger.error(f"NaN or Inf detected in logs for batch {batch}!")
                    for k, v in logs.items():
                        if np.isnan(v) or np.isinf(v):
                            logger.error(f"  - {k} = {v}")
        
        # Add NaN checking callback
        nan_callback = NanInfCallback()
        callbacks.append(nan_callback)
        
        # Start training with error handling
        start_time = time.time()
        
        try:
            # Train the model
            history = self.model.fit(
                train_dataset,
                epochs=epochs,
                initial_epoch=initial_epoch,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_dataset,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            # Training completed
            training_time = time.time() - start_time
            hours, remainder = divmod(training_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Generate final visualizations
            if visualize and validation_dataset:
                self.generate_visualizations(validation_dataset, "final")
            
            # Save final model
            final_path = os.path.join(LOCAL_TEMP_DIR, "checkpoints", "prism_final_model.h5")
            self.model.save(final_path)
            
            # Upload to GCS
            final_gcs_path = f"{GCS_CHECKPOINT_DIR}/prism_final_model.h5"
            blob = bucket.blob(final_gcs_path)
            blob.upload_from_filename(final_path)
            
            return history
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            
            # Try to save model snapshot at failure point
            try:
                error_path = os.path.join(LOCAL_TEMP_DIR, "checkpoints", "prism_error_snapshot.h5")
                self.model.save_weights(error_path)
                
                error_gcs_path = f"{GCS_CHECKPOINT_DIR}/prism_error_snapshot.h5"
                blob = bucket.blob(error_gcs_path)
                blob.upload_from_filename(error_path)
            except:
                logger.error("Failed to save model snapshot at error point")
            
            return None
# =========================
# Main Training Function
# =========================

def main():
    """Main training function for PRISM model."""
    # Print script information and configuration
    logger.info("PRISM Module Training Script")
    
    # Find available samples for training and validation
    try:
        train_samples = find_matching_samples(training=True)
        if args.use_validation:
            val_samples = find_matching_samples(training=False)
        else:
            val_samples = []
    except Exception as e:
        logger.error(f"Error finding samples: {e}")
        sys.exit(1)
    
    logger.info(f"Found {len(train_samples)} training samples and {len(val_samples)} validation samples")
    
    # Create datasets
    try:
        train_dataset = create_prism_dataset(
            triplets=train_samples, 
            batch_size=args.batch_size, 
            training=True
        )
        
        if args.use_validation:
            val_dataset = create_prism_dataset(
                triplets=val_samples, 
                batch_size=args.val_batch_size, 
                training=False
            )
        else:
            val_dataset = None
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        sys.exit(1)
    
    # Calculate steps per epoch
    steps_per_epoch = max(1, len(train_samples) // args.batch_size)
    if args.use_validation:
        validation_steps = max(1, len(val_samples) // args.val_batch_size)
    else:
        validation_steps = None
    
    # Build the PRISM model
    try:
        # Use distributed strategy if multi-GPU is enabled
        if args.multi_gpu:
            try:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    prism_model = SimplePRISMModel(
                        input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3),
                        lithos_shape=(LITHOS_HEIGHT, LITHOS_WIDTH, LITHOS_CHANNELS),
                        training=True,
                        learning_rate=args.learning_rate,
                        dropout_rate=args.dropout_rate,
                        l2_reg=args.l2_reg,
                        use_attention=args.use_attention,
                        clothing_weight=args.clothing_weight,
                        background_weight=args.background_weight,
                        perceptual_weight=args.perceptual_weight,
                        gradient_clip=args.gradient_clip
                    )
            except Exception:
                args.multi_gpu = False
        
        # Fall back to single-GPU if multi-GPU is not enabled or failed
        if not args.multi_gpu:
            prism_model = SimplePRISMModel(
                input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3),
                lithos_shape=(LITHOS_HEIGHT, LITHOS_WIDTH, LITHOS_CHANNELS),
                training=True,
                learning_rate=args.learning_rate,
                dropout_rate=args.dropout_rate,
                l2_reg=args.l2_reg,
                use_attention=args.use_attention,
                clothing_weight=args.clothing_weight,
                background_weight=args.background_weight,
                perceptual_weight=args.perceptual_weight,
                gradient_clip=args.gradient_clip
            )
    except Exception as e:
        logger.error(f"Error building PRISM model: {e}")
        sys.exit(1)
    
    # Load checkpoint if specified
    if args.load_checkpoint or args.load_step:
        loaded = prism_model.load_checkpoint(
            checkpoint_path=args.load_checkpoint,
            step=args.load_step
        )
        if loaded:
            logger.info("Successfully loaded checkpoint")
        else:
            logger.warning("Failed to load checkpoint, starting with fresh weights")
    
    # Print model summary
    prism_model.model.summary(print_fn=logger.info)
    
    # Generate initial visualizations if debugging is enabled
    if args.debug_visualize:
        try:
            if val_dataset:
                prism_model.generate_visualizations(val_dataset, "initial")
            else:
                prism_model.generate_visualizations(train_dataset, "initial")
        except Exception as e:
            logger.warning(f"Failed to generate initial visualizations: {e}")
    
    # Train the model
    try:
        history = prism_model.train(
            train_dataset=train_dataset,
            validation_dataset=val_dataset if args.use_validation else None,
            epochs=args.epochs,
            initial_epoch=args.initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            checkpoint_steps=args.checkpoint_steps,
            visualize=args.debug_visualize
        )
        
        if history:
            logger.info("PRISM model training completed successfully!")
        else:
            logger.error("PRISM model training failed")
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        
        # Try to save model snapshot at failure point
        try:
            logger.info("Attempting to save model snapshot at error point")
            error_path = os.path.join(LOCAL_TEMP_DIR, "checkpoints", "prism_error_snapshot.h5")
            prism_model.model.save_weights(error_path)
            
            error_gcs_path = f"{GCS_CHECKPOINT_DIR}/prism_error_snapshot.h5"
            blob = bucket.blob(error_gcs_path)
            blob.upload_from_filename(error_path)
        except Exception:
            logger.error("Failed to save model snapshot at error point")

if __name__ == "__main__":
    # Set memory growth for GPUs if available
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass
    
    # Import matplotlib if visualization is enabled
    if args.debug_visualize:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            args.debug_visualize = False
    
    # Run the main function
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        sys.exit(1)