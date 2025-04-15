#!/usr/bin/env python3
# LITHOS Training Script for Vertex AI with Multi-View Support and Enhanced FSE
# ============================================================

import os
import sys
import time
import argparse
import logging
import json
import io
from datetime import datetime
import random
import h5py

import numpy as np
import tensorflow as tf
from google.cloud import storage
import cv2

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Dropout,
    GlobalAveragePooling2D, Dense, UpSampling2D, Concatenate, Resizing
)
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train LITHOS module on Vertex AI')
    parser.add_argument('--project-id', type=str, default='bright-link-455716')
    parser.add_argument('--bucket-name', type=str, default='auralith')
    parser.add_argument('--base-path', type=str, default='lithos')
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--val-batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--initial-epoch', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=50000)
    parser.add_argument('--max-val-samples', type=int, default=5000)
    parser.add_argument('--checkpoint-steps', type=int, default=500)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    parser.add_argument('--l2-reg', type=float, default=1e-5)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--base-encoding-weight', type=float, default=1.0)
    parser.add_argument('--material-class-weight', type=float, default=0.5)
    parser.add_argument('--importance-weight', type=float, default=0.3)  # Added for importance map weight
    parser.add_argument('--local-tmp-dir', type=str, default='/tmp/auralith')
    parser.add_argument('--use-validation', action='store_true')
    parser.add_argument('--skip-checkpoint', action='store_true')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'efficientnet_b0', 'custom'])
    parser.add_argument('--input-size', type=int, default=1024)
    parser.add_argument('--multi-view', action='store_true')
    parser.add_argument('--num-view-angles', type=int, default=8)
    parser.add_argument('--use-enhanced-augmentation', action='store_true')  # Flag for enhanced augmentations
    parser.add_argument('--use-importance-map', action='store_true')  # Flag for importance-based quantization
    parser.add_argument('--load-step', type=int, default=None, help='Specific checkpoint step to load (e.g., 2500)')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Specific checkpoint path to load')
    
    # New arguments for enhanced architecture
    parser.add_argument('--fse-channels', type=int, default=6, 
                       help="Number of channels in the FSE encoding (default: 6)")
    parser.add_argument('--use-decoder-supervision', action='store_true',
                       help="Add decoder branch during training for supervision")
    parser.add_argument('--decoded-rgb-weight', type=float, default=2.0,
                       help="Weight for decoded RGB loss (if decoder supervision enabled)")
    parser.add_argument('--distribution-weight', type=float, default=0.05,
                       help="Weight for distribution regularization loss")
    
    return parser.parse_args()

args = parse_args()
# Global constants from args
PROJECT_ID = args.project_id
GCS_BUCKET_NAME = args.bucket_name
GCS_BASE_PATH = args.base_path
LOCAL_TEMP_DIR = args.local_tmp_dir
INPUT_SIZE = args.input_size

# Define GCS paths
GCS_MATERIAL_IMAGES_PATH = f"{GCS_BASE_PATH}/material_images"
GCS_REFERENCE_ENCODING_PATH = f"{GCS_BASE_PATH}/reference_encodings"
GCS_MATERIAL_METADATA_PATH = f"{GCS_BASE_PATH}/material_metadata"
GCS_CHECKPOINT_DIR = f"{GCS_BASE_PATH}/checkpoints"

# Validation paths
GCS_VAL_MATERIAL_IMAGES_PATH = f"{GCS_BASE_PATH}/val/material_images"
GCS_VAL_REFERENCE_ENCODING_PATH = f"{GCS_BASE_PATH}/val/reference_encodings"
GCS_VAL_MATERIAL_METADATA_PATH = f"{GCS_BASE_PATH}/val/material_metadata"

# Supported material classes
MATERIAL_CLASSES = [
    'cotton', 'polyester', 'leather', 'denim', 'silk', 'wool',
    'linen', 'nylon', 'metal', 'plastic', 'glass', 'ceramic',
    'wood', 'rubber', 'velvet', 'canvas'
]
NUM_MATERIAL_CLASSES = len(MATERIAL_CLASSES)

# Ensure temp directory exists
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

# Initialize GCS bucket
try:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    logger.info(f"✅ Connected to GCS bucket '{GCS_BUCKET_NAME}'")
except Exception as e:
    logger.error(f"❌ Failed to connect to GCS: {e}")
    sys.exit(1)
    
# === Enhanced Augmentation Functions ===

def apply_directional_lighting(image, direction=[0.5, -0.7, 0.2], intensity=0.3):
    """Apply directional lighting to simulate different lighting conditions."""
    height, width = image.shape[0], image.shape[1]
    
    # Create a gradient based on direction vector
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    gradient = x * direction[0] + y * direction[1] + direction[2]
    gradient = (gradient + 1) / 2.0  # Normalize to 0-1
    
    # Convert to tensor with proper dimensions
    gradient_tensor = tf.convert_to_tensor(gradient, dtype=tf.float32)
    gradient_tensor = tf.expand_dims(gradient_tensor, axis=-1)
    gradient_tensor = tf.repeat(gradient_tensor, 3, axis=-1)
    
    # Apply lighting as a multiplicative effect
    lit_image = image * (1.0 + intensity * (gradient_tensor - 0.5))
    lit_image = tf.clip_by_value(lit_image, 0.0, 1.0)
    
    return lit_image

def adjust_exposure(image, factor=0.7):
    """Adjust image exposure to simulate different lighting intensities."""
    adjusted = image * factor
    adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    return adjusted

def enhance_material_property(image, property='metalness', intensity=0.3):
    """Enhance a specific material property in the image."""
    if property == 'metalness':
        # Increase contrast and add slight blue tint for metallic appearance
        enhanced = tf.image.adjust_contrast(image, 1.5)
        # Add slight color shift toward metallic blue
        blue_shift = tf.ones_like(image) * tf.constant([0.0, 0.0, 0.1], dtype=tf.float32)
        enhanced = enhanced + blue_shift * intensity
    elif property == 'roughness':
        # Decrease contrast and add noise for roughness
        enhanced = tf.image.adjust_contrast(image, 0.8)
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
        enhanced = enhanced + noise * intensity
    else:
        enhanced = image
        
    enhanced = tf.clip_by_value(enhanced, 0.0, 1.0)
    return enhanced

def enhance_training_data(image_batch):
    """Apply enhanced augmentations to a batch of images."""
    if not args.use_enhanced_augmentation:
        return image_batch
        
    # Create a larger batch with variations
    images = []
    for i in range(tf.shape(image_batch)[0]):
        image = image_batch[i]
        images.append(image)  # Original image
        
        # Only augment some of the images to maintain a balance
        if tf.random.uniform(()) > 0.7:
            # Create lighting variations with random parameters
            direction = [
                tf.random.uniform((), -1.0, 1.0),
                tf.random.uniform((), -1.0, 1.0),
                tf.random.uniform((), 0.0, 0.5)
            ]
            intensity = tf.random.uniform((), 0.1, 0.4)
            
            lit_image = apply_directional_lighting(image, direction=direction, intensity=intensity)
            images.append(lit_image)
            
        if tf.random.uniform(()) > 0.7:
            # Adjust exposure with random factor
            factor = tf.random.uniform((), 0.6, 1.4)
            darkened_image = adjust_exposure(image, factor=factor)
            images.append(darkened_image)
            
        if tf.random.uniform(()) > 0.8:
            # Enhance material properties randomly
            prop = 'metalness' if tf.random.uniform(()) > 0.5 else 'roughness'
            intensity = tf.random.uniform((), 0.2, 0.5)
            mat_enhanced = enhance_material_property(image, property=prop, intensity=intensity)
            images.append(mat_enhanced)
    
    # Return a random subset to maintain the original batch size
    images_tensor = tf.stack(images)
    indices = tf.random.shuffle(tf.range(tf.shape(images_tensor)[0]))[:tf.shape(image_batch)[0]]
    return tf.gather(images_tensor, indices)

# === Data Listing and Downloading ===

def list_available_samples(prefix_path, file_extension='.jpg', list_all=False):
    """List available sample files with a specific extension in a GCS prefix."""
    try:
        logger.info(f"📂 Scanning {prefix_path} for '*{file_extension}'...")
        blobs = list(bucket.list_blobs(prefix=f"{prefix_path}/"))
        files = {
            os.path.splitext(os.path.basename(blob.name))[0]
            for blob in blobs if blob.name.endswith(file_extension)
        }
        logger.info(f"📁 Found {len(files)} files in {prefix_path}")
        return list(files) if list_all else files
    except Exception as e:
        logger.error(f"❌ Error listing samples from {prefix_path}: {e}")
        return set()

def find_common_samples(training=True):
    """Find samples with just images and metadata (no .npy required)."""
    img_path = GCS_MATERIAL_IMAGES_PATH if training else GCS_VAL_MATERIAL_IMAGES_PATH
    meta_path = GCS_MATERIAL_METADATA_PATH if training else GCS_VAL_MATERIAL_METADATA_PATH

    material_images = list_available_samples(img_path, '.jpg') | list_available_samples(img_path, '.png')
    metadata_files = list_available_samples(meta_path, '.json')

    common = material_images & metadata_files
    logger.info(f"✅ Found {len(common)} samples with image + metadata (no .npy)")
    return list(common)

def download_material_image_from_gcs(image_id, training=True):
    """Download image from GCS and return as Tensor."""
    prefix = GCS_MATERIAL_IMAGES_PATH if training else GCS_VAL_MATERIAL_IMAGES_PATH
    for ext in ['.jpg', '.png']:
        try:
            blob_path = f"{prefix}/{image_id}{ext}"
            blob = bucket.blob(blob_path)
            if blob.exists():
                image_bytes = blob.download_as_bytes()
                img = tf.image.decode_image(image_bytes, channels=3)
                img = tf.image.resize(img, (INPUT_SIZE, INPUT_SIZE))
                return tf.cast(img, tf.float32) / 255.0
        except Exception as e:
            logger.warning(f"⚠️ Could not load image {image_id}{ext}: {e}")
    return tf.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32)

def load_reference_encoding_from_gcs(image_id, training=True):
    """Load 10-bit encoding from GCS and return as Tensor."""
    prefix = GCS_REFERENCE_ENCODING_PATH if training else GCS_VAL_REFERENCE_ENCODING_PATH
    try:
        blob = bucket.blob(f"{prefix}/{image_id}.npy")
        if blob.exists():
            raw = blob.download_as_bytes()
            encoding = np.load(io.BytesIO(raw))
            if encoding.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
                encoding = cv2.resize(encoding, (INPUT_SIZE, INPUT_SIZE))
            if encoding.ndim == 2:
                encoding = np.stack([encoding] * 3, axis=-1)
            elif encoding.shape[-1] == 4:
                encoding = encoding[..., :3]
            return tf.convert_to_tensor(encoding, dtype=tf.float32)
    except Exception as e:
        logger.warning(f"⚠️ Failed to load encoding for {image_id}: {e}")
    return tf.ones((INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32) * 0.5

# === Metadata Loading and Multi-View Utilities ===

def load_material_metadata_from_gcs(image_id, training=True):
    """Load material metadata and return class one-hot + material property tensor."""
    prefix = GCS_MATERIAL_METADATA_PATH if training else GCS_VAL_MATERIAL_METADATA_PATH
    try:
        blob = bucket.blob(f"{prefix}/{image_id}.json")
        if not blob.exists():
            raise FileNotFoundError(f"Metadata not found for {image_id}")

        metadata = json.loads(blob.download_as_string())
        material_class = metadata.get('material_class', 'unknown')
        class_index = MATERIAL_CLASSES.index(material_class) if material_class in MATERIAL_CLASSES else -1
        one_hot = tf.one_hot(class_index, NUM_MATERIAL_CLASSES) if class_index >= 0 else tf.zeros(NUM_MATERIAL_CLASSES)

        props = metadata.get('properties', {})
        properties_tensor = tf.convert_to_tensor([
            float(props.get('reflectivity', 0.5)),
            float(props.get('roughness', 0.5)),
            float(props.get('metalness', 0.0)),
            float(props.get('transparency', 0.0))
        ], dtype=tf.float32)

        return one_hot, properties_tensor
    except Exception as e:
        logger.warning(f"⚠️ Metadata load failed for {image_id}: {e}")
        return tf.zeros(NUM_MATERIAL_CLASSES), tf.convert_to_tensor([0.5, 0.5, 0.0, 0.0], dtype=tf.float32)

def load_product_metadata_from_gcs(product_id, training=True):
    """Load full product metadata including multiple views."""
    prefix = GCS_MATERIAL_METADATA_PATH if training else GCS_VAL_MATERIAL_METADATA_PATH
    try:
        blob = bucket.blob(f"{prefix}/{product_id}.json")
        return json.loads(blob.download_as_string()) if blob.exists() else None
    except Exception as e:
        logger.warning(f"⚠️ Error loading product metadata for {product_id}: {e}")
        return None

def find_related_views(product_metadata):
    """Get available views for a given product."""
    if not product_metadata or 'views' not in product_metadata:
        return []
    return product_metadata['views']

def create_view_angle_encoding(view_angle, num_angles=8):
    """One-hot encode a view angle."""
    bin_idx = round(view_angle / (360 / num_angles)) % num_angles
    return tf.one_hot(bin_idx, num_angles)

def load_all_product_views(product_metadata, training=True):
    """Return list of views with images and view angle encodings."""
    views = []
    for view in product_metadata.get('views', []):
        filename = view.get('image_filename')
        if not filename:
            continue
        image_id = os.path.splitext(filename)[0]
        try:
            view_data = {
                'image_id': image_id,
                'view_id': view.get('view_id', 'unknown'),
                'view_angle': view.get('view_angle', 0),
                'primary_view': view.get('primary_view', False),
                'image': download_material_image_from_gcs(image_id, training),
                'angle_encoding': create_view_angle_encoding(view.get('view_angle', 0))
            }
            views.append(view_data)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load view {image_id}: {e}")
            continue
    return views

def find_multi_view_products(training=True):
    """Return product IDs that have multi-view data."""
    prefix = GCS_MATERIAL_METADATA_PATH if training else GCS_VAL_MATERIAL_METADATA_PATH
    try:
        blobs = list(bucket.list_blobs(prefix=f"{prefix}/"))
        product_ids = set()
        for blob in blobs:
            if blob.name.endswith('.json'):
                try:
                    meta = json.loads(blob.download_as_string())
                    if meta.get('views'):
                        pid = meta.get('product_id')
                        if pid:
                            product_ids.add(pid)
                except Exception as e:
                    logger.warning(f"⚠️ Could not read metadata in {blob.name}: {e}")
        return list(product_ids)
    except Exception as e:
        logger.error(f"❌ Failed to find multi-view products: {e}")
        return []

# === Augmentation and Standard Dataset Creation ===

def apply_augmentation(image, reference_encoding):
    """Apply consistent random augmentations to image and encoding."""
    if tf.random.uniform(()) > 0.5:
        flip = tf.random.uniform(()) > 0.5
        if flip:
            image = tf.image.flip_left_right(image)
            reference_encoding = tf.image.flip_left_right(reference_encoding)

        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)

    return image, reference_encoding

def data_generator(sample_ids, batch_size=16, training=True, repeat=True):
    """Standard single-view data generator."""
    samples = sample_ids.copy()

    def process_epoch():
        if training:
            random.shuffle(samples)

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batch_images, batch_encodings, batch_classes, batch_props = [], [], [], []

            for img_id in batch:
                try:
                    image = download_material_image_from_gcs(img_id, training)
                    reference = image  # ⬅️ Use image as base_encoding label (self-supervised)
                    mat_class, mat_props = load_material_metadata_from_gcs(img_id, training)

                    if training:
                        image, reference = apply_augmentation(image, reference)

                    batch_images.append(image)
                    batch_encodings.append(reference)
                    batch_classes.append(mat_class)
                    batch_props.append(mat_props)
                except Exception as e:
                    logger.warning(f"⚠️ Skipped sample {img_id} due to error: {e}")
                    continue

            if not batch_images:
                continue

            batch_x = tf.stack(batch_images)
            
            # Apply enhanced augmentations if enabled
            if training and args.use_enhanced_augmentation:
                batch_x = enhance_training_data(batch_x)
                
            batch_y = {
                'lithos_base_encoding': tf.stack(batch_encodings),
                'lithos_material_class': tf.stack(batch_classes),
                'lithos_material_properties': tf.stack(batch_props)
            }
            
            # Add decoded RGB target if using decoder supervision
            if args.use_decoder_supervision:
                batch_y['decoded_rgb'] = tf.stack(batch_encodings)  # Same as the original image
                
            if args.use_importance_map:
                importance_stack = tf.stack([
                    tf.ones_like(batch_encodings[0]) for _ in range(len(batch_encodings))
                ])
                batch_y['channel_importance'] = importance_stack
            

            yield batch_x, batch_y

    while True:
        yield from process_epoch()
        if not repeat:
            break

def create_lithos_dataset(sample_ids, batch_size=16, training=True):
    """Create TensorFlow dataset for standard (single-view) LITHOS training."""
    # Base output keys
    output_keys = {
        'lithos_base_encoding': tf.TensorSpec(shape=(None, INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32),
        'lithos_material_class': tf.TensorSpec(shape=(None, NUM_MATERIAL_CLASSES), dtype=tf.float32),
        'lithos_material_properties': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    }

    # Add decoder output if using decoder supervision
    if args.use_decoder_supervision:
        output_keys['decoded_rgb'] = tf.TensorSpec(shape=(None, INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32)

    if args.use_importance_map:
        output_keys['channel_importance'] = tf.TensorSpec(shape=(None, INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32)

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(sample_ids, batch_size, training, repeat=True),
        output_signature=(
            tf.TensorSpec(shape=(None, INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32),
            output_keys
        )
    )
    return dataset.prefetch(tf.data.AUTOTUNE)

# === Multi-View Dataset Generator ===

def multi_view_data_generator(product_ids, batch_size=16, training=True, repeat=True):
    """Yield batches of multi-view product data."""
    products = product_ids.copy()

    def process_epoch():
        if training:
            random.shuffle(products)

        for i in range(0, len(products), batch_size):
            batch_ids = products[i:i+batch_size]
            batch_images, batch_angles, batch_classes, batch_props = [], [], [], []

            for pid in batch_ids:
                try:
                    metadata = load_product_metadata_from_gcs(pid, training)
                    if not metadata:
                        continue

                    views = load_all_product_views(metadata, training)
                    if not views:
                        continue

                    # Select primary or random view
                    view = random.choice(views) if training else next(
                        (v for v in views if v.get('primary_view')), views[0])

                    image = view['image']
                    angle_encoding = view['angle_encoding']

                    if training:
                        image = tf.image.random_brightness(image, 0.1)
                        image = tf.image.random_contrast(image, 0.9, 1.1)
                        image = tf.image.random_saturation(image, 0.8, 1.2)
                        image = tf.clip_by_value(image, 0.0, 1.0)

                    mat_class = metadata.get('material_class', 'unknown')
                    class_idx = MATERIAL_CLASSES.index(mat_class) if mat_class in MATERIAL_CLASSES else -1
                    one_hot = tf.one_hot(class_idx, NUM_MATERIAL_CLASSES) if class_idx >= 0 else tf.zeros(NUM_MATERIAL_CLASSES)

                    props = metadata.get('properties', {})
                    prop_tensor = tf.convert_to_tensor([
                        float(props.get('reflectivity', 0.5)),
                        float(props.get('roughness', 0.5)),
                        float(props.get('metalness', 0.0)),
                        float(props.get('transparency', 0.0))
                    ], dtype=tf.float32)

                    batch_images.append(image)
                    batch_angles.append(angle_encoding)
                    batch_classes.append(one_hot)
                    batch_props.append(prop_tensor)
                except Exception as e:
                    logger.warning(f"⚠️ Error in multi-view sample {pid}: {e}")
                    continue

            if not batch_images:
                continue

            # Stack and apply enhanced augmentations if enabled
            stacked_images = tf.stack(batch_images)
            if training and args.use_enhanced_augmentation:
                stacked_images = enhance_training_data(stacked_images)
                
            batch_x = {
                'image_input': stacked_images,
                'view_angle_input': tf.stack(batch_angles)
            }

            batch_y = {
                'lithos_base_encoding': tf.stack(batch_images),  # Self-supervised
                'lithos_material_class': tf.stack(batch_classes),
                'lithos_material_properties': tf.stack(batch_props)
            }
            
            # Add decoded RGB output if using decoder supervision
            if args.use_decoder_supervision:
                batch_y['decoded_rgb'] = tf.stack(batch_images)  # Same as input

            yield batch_x, batch_y

    while True:
        yield from process_epoch()
        if not repeat:
            break

def create_multi_view_lithos_dataset(product_ids, batch_size=16, training=True):
    """Create a TensorFlow dataset for multi-view training."""
    # Base output structure
    outputs = {
        'lithos_base_encoding': tf.TensorSpec(shape=(None, INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32),
        'lithos_material_class': tf.TensorSpec(shape=(None, NUM_MATERIAL_CLASSES), dtype=tf.float32),
        'lithos_material_properties': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    }
    
    # Add decoder output if using supervision
    if args.use_decoder_supervision:
        outputs['decoded_rgb'] = tf.TensorSpec(shape=(None, INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32)
    
    dataset = tf.data.Dataset.from_generator(
        lambda: multi_view_data_generator(product_ids, batch_size, training),
        output_signature=(
            {
                'image_input': tf.TensorSpec(shape=(None, INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32),
                'view_angle_input': tf.TensorSpec(shape=(None, args.num_view_angles), dtype=tf.float32),
            },
            outputs
        )
    )
    return dataset.prefetch(tf.data.AUTOTUNE)

# After all your data loading and preprocessing functions
# After the create_multi_view_lithos_dataset function
# Before the LithosModel class definition

def split_train_val_data(sample_ids, val_ratio=0.15):
    """Split data into training and validation sets."""
    # Shuffle first to ensure random split
    shuffled_samples = sample_ids.copy()
    random.shuffle(shuffled_samples)
    
    # Calculate the split index
    val_count = int(len(shuffled_samples) * val_ratio)
    
    # Split the data
    val_samples = shuffled_samples[:val_count]
    train_samples = shuffled_samples[val_count:]
    
    logger.info(f"Split {len(sample_ids)} samples into {len(train_samples)} training and {len(val_samples)} validation")
    return train_samples, val_samples


# === Neural Network Architecture ===

class LithosModel:
    """
    Enhanced LITHOS Model with:
    - Expanded float space (6 channels by default)
    - Decoder supervision
    - Distribution regularization
    - Color fidelity optimization
    """

    def __init__(self,
                 input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
                 backbone_type="mobilenet_v2",
                 training=False,
                 learning_rate=0.001,
                 dropout_rate=0.1,
                 l2_reg=1e-6,
                 loss_weights=None,
                 use_importance_map=False,
                 fse_channels=6,
                 use_decoder_supervision=False):
        
        self.input_shape = input_shape
        self.backbone_type = backbone_type
        self.training = training
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_importance_map = use_importance_map
        self.fse_channels = fse_channels
        self.use_decoder_supervision = use_decoder_supervision
        self.loss_weights = loss_weights or {
            "lithos_base_encoding": 1.5,
            "lithos_material_class": 0.2,
            "lithos_material_properties": 0.1,
            "decoded_rgb": 2.0 if use_decoder_supervision else 0.0
        }
        
        # Add importance map weight if enabled
        if self.use_importance_map:
            self.loss_weights["channel_importance"] = args.importance_weight

        self.skip_features = {}
        self._build_model()

        if training:
            self._compile_model()

    def _build_model(self):
        """Build LITHOS model architecture with decoder supervision and expanded channels."""
        # Input layer
        inputs = Input(shape=self.input_shape, name="image_input")
        
        # Build backbone and extract features
        backbone_output = self._build_backbone(inputs)

        # Build the expanded base encoding branch with fse_channels
        base_encoding = self._build_base_encoding_branch(backbone_output)
        
        # Material classification and properties branches
        material_class = self._build_material_class_branch(backbone_output)
        material_properties = self._build_material_properties_branch(backbone_output)
        
        # Create outputs dictionary
        outputs = {
            "lithos_base_encoding": base_encoding,
            "lithos_material_class": material_class,
            "lithos_material_properties": material_properties
        }
        
        # Add decoder supervision branch for training
        if self.training and self.use_decoder_supervision:
            decoded_rgb = self._build_decoder_branch(base_encoding)
            outputs["decoded_rgb"] = decoded_rgb
        
        # Add importance map branch if enabled
        if self.use_importance_map:
            importance_map = self._build_importance_map_branch(backbone_output)
            outputs["channel_importance"] = importance_map
        
        # Create the model
        self.model = Model(inputs=inputs, outputs=outputs, name="lithos_model")

    def _build_backbone(self, inputs):
        """Construct feature extractor backbone with skip connections (functional-safe)."""
        if self.backbone_type == "mobilenet_v2":
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet",
                alpha=0.75
            )
            base_model.trainable = self.training

            # Collect required outputs from base_model
            layer_map = {
                'level_1': 'expanded_conv_project_BN',
                'level_2': 'block_3_expand_relu',
                'level_3': 'block_6_expand_relu',
                'level_4': 'block_13_expand_relu'
            }

            outputs = {}
            for key, layer_name in layer_map.items():
                try:
                    outputs[key] = base_model.get_layer(layer_name).output
                except:
                    logger.warning(f"⚠️ Could not get skip layer: {layer_name}")

            outputs['final'] = base_model.output

            # Rebuild base model with functional graph from input to outputs
            model_with_skips = Model(inputs=base_model.input, outputs=outputs)
            result = model_with_skips(inputs)

            # Store skip connections
            self.skip_features = {k: result[k] for k in outputs if k != 'final'}
            return result['final']

        elif self.backbone_type == "efficientnet_b0":
            base_model = EfficientNetB0(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet"
            )
            base_model.trainable = self.training

            outputs = {
                'level_1': base_model.get_layer('block1a_project_bn').output,
                'level_2': base_model.get_layer('block2b_add').output,
                'level_3': base_model.get_layer('block3b_add').output,
                'level_4': base_model.get_layer('block5c_add').output,
                'final': base_model.output
            }

            model_with_skips = Model(inputs=base_model.input, outputs=outputs)
            result = model_with_skips(inputs)

            self.skip_features = {k: result[k] for k in outputs if k != 'final'}
            return result['final']

        else:
            return self._build_custom_backbone(inputs)

    def _build_custom_backbone(self, inputs):
        """Fallback custom lightweight backbone."""
        x = Conv2D(32, 3, strides=2, padding='same', kernel_regularizer=regularizers.l2(self.l2_reg))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        self.skip_features['level_1'] = x

        x = self._inverted_res_block(x, filters=64, strides=2, expansion=6, block_id=1)
        self.skip_features['level_2'] = x

        x = self._inverted_res_block(x, filters=96, strides=2, expansion=6, block_id=2)
        self.skip_features['level_3'] = x

        x = self._inverted_res_block(x, filters=160, strides=2, expansion=6, block_id=3)
        self.skip_features['level_4'] = x

        return x
        
    def _build_importance_map_branch(self, backbone_features):
        """Build importance map prediction branch for precision-guided encoding."""
        with tf.name_scope('importance_map_branch'):
            x = Conv2D(64, kernel_size=1, padding='same',
                      kernel_regularizer=regularizers.l2(self.l2_reg),
                      name='importance_reduce_conv')(backbone_features)
            x = BatchNormalization(name='importance_reduce_bn')(x)
            x = Activation('relu', name='importance_reduce_relu')(x)
            x = Dropout(self.dropout_rate, name='importance_dropout')(x)
            
            # Use same upsampling path as the base encoding branch
            input_shape = self.input_shape
            skip_names = ['level_4', 'level_3', 'level_2', 'level_1']
            skip_filters = [64, 48, 32, 24]

            for i, (level, filters) in enumerate(zip(skip_names, skip_filters), start=1):
                x = UpSampling2D(size=(2, 2), name=f'importance_upsample{i}')(x)

                if level in self.skip_features:
                    skip = self.skip_features[level]
                    skip = Conv2D(filters, kernel_size=1, padding='same', name=f'importance_skip{i}_proj')(skip)

                    # Resize skip to match current upsampled x shape
                    target_height = x.shape[1] if x.shape[1] is not None else input_shape[0]
                    target_width = x.shape[2] if x.shape[2] is not None else input_shape[1]
                    skip = Resizing(target_height, target_width, name=f'importance_skip{i}_resize')(skip)

                    x = Concatenate(name=f'importance_skip_concat{i}')([x, skip])

                x = Conv2D(filters, kernel_size=3, padding='same',
                          kernel_regularizer=regularizers.l2(self.l2_reg),
                          name=f'importance_conv{i}')(x)
                x = BatchNormalization(name=f'importance_bn{i}')(x)
                x = Activation('relu', name=f'importance_relu{i}')(x)

            # Final upsampling to restore full resolution
            x = UpSampling2D(size=(2, 2), name='importance_upsample_final')(x)
            
            # Generate per-channel importance maps (match channels to fse_channels)
            channels = min(3, self.fse_channels) # Only support up to 3 channels for importance map for now
            importance_map = Conv2D(channels, kernel_size=1, activation='sigmoid',
                                  kernel_regularizer=regularizers.l2(self.l2_reg),
                                  name='channel_importance')(x)

            return importance_map

    def _build_base_encoding_branch(self, backbone_features):
        """Build base encoding branch with expanded channel output based on fse_channels."""
        with tf.name_scope('base_encoding_branch'):
            x = Conv2D(128, kernel_size=1, padding='same',
                   kernel_regularizer=regularizers.l2(self.l2_reg),
                   name='encoding_reduce_conv')(backbone_features)
            x = BatchNormalization(name='encoding_reduce_bn')(x)
            x = Activation('relu', name='encoding_reduce_relu')(x)
            x = Dropout(self.dropout_rate, name='encoding_dropout')(x)

            input_shape = self.input_shape
            skip_names = ['level_4', 'level_3', 'level_2', 'level_1']
            skip_filters = [128, 96, 64, 48]

            for i, (level, filters) in enumerate(zip(skip_names, skip_filters), start=1):
                x = UpSampling2D(size=(2, 2), name=f'encoding_upsample{i}')(x)

                if level in self.skip_features:
                    skip = self.skip_features[level]
                    skip = Conv2D(filters, kernel_size=1, padding='same', name=f'skip{i}_proj')(skip)

                    # Resize skip to match current upsampled x shape
                    target_height = x.shape[1] if x.shape[1] is not None else input_shape[0]
                    target_width = x.shape[2] if x.shape[2] is not None else input_shape[1]
                    skip = Resizing(target_height, target_width, name=f'skip{i}_resize')(skip)

                    x = Concatenate(name=f'skip_concat{i}')([x, skip])

                x = Conv2D(filters, kernel_size=3, padding='same',
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        name=f'encoding_conv{i}')(x)
                x = BatchNormalization(name=f'encoding_bn{i}')(x)
                x = Activation('relu', name=f'encoding_relu{i}')(x)

            x = Conv2D(32, kernel_size=3, padding='same',
                    kernel_regularizer=regularizers.l2(self.l2_reg),
                    name='encoding_refine1')(x)
            x = BatchNormalization(name='encoding_refine_bn1')(x)
            x = Activation('relu', name='encoding_refine_relu1')(x)
            
            # Final upsampling to restore full resolution
            x = UpSampling2D(size=(2, 2), name='encoding_upsample_final')(x)

            # Use the fse_channels parameter for number of output channels
            base_encoding = Conv2D(self.fse_channels, kernel_size=1, activation='sigmoid',
                                kernel_regularizer=regularizers.l2(self.l2_reg),
                                name='lithos_base_encoding')(x)

            return base_encoding

    def _build_decoder_branch(self, base_encoding):
        """Build PRISM-like decoder for supervising base encoding quality."""
        with tf.name_scope('decoder_branch'):
            # Simple decoder that takes expanded base_encoding and outputs RGB (3 channels)
            x = Conv2D(64, 3, padding='same', activation='relu', 
                      kernel_regularizer=regularizers.l2(self.l2_reg),
                      name='decoder_conv1')(base_encoding)
            
            # Add some depth for better decoding
            x = Conv2D(48, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(self.l2_reg),
                      name='decoder_conv2')(x)
            
            # Final layer to produce RGB output
            decoded_rgb = Conv2D(3, 3, padding='same', activation='sigmoid',
                                kernel_regularizer=regularizers.l2(self.l2_reg),
                                name='decoded_rgb')(x)
            
            return decoded_rgb

    def _build_material_class_branch(self, backbone_features):
        """Branch for material classification."""
        x = GlobalAveragePooling2D()(backbone_features)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        return Dense(NUM_MATERIAL_CLASSES, activation='softmax', name='lithos_material_class')(x)

    def _build_material_properties_branch(self, backbone_features):
        """Branch for predicting material physical properties."""
        x = GlobalAveragePooling2D()(backbone_features)
        x = Dense(32, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        return Dense(4, activation='sigmoid', name='lithos_material_properties')(x)

    def _compile_model(self):
        """Compile the model with comprehensive loss functions including distribution regularization."""
        
        # === Float distribution regularization loss ===
        # === Fixed Float distribution regularization loss ===
        def float_distribution_loss(y_true, y_pred):
            """Ensure float values are well-distributed with entropy regularization."""
            # Add small epsilon to avoid log(0)
            epsilon = 1e-7
            
            # Calculate histogram with fixed bins for stability
            nbins = 20
            target_dist = tf.ones([nbins]) / nbins  # Uniform target distribution
            
            # Get the number of channels as a static value if possible
            # This fixes the 'SymbolicTensor' object cannot be interpreted as an integer error
            try:
                # Try to get the static shape
                num_channels = y_pred.shape[-1]
                if num_channels is None:
                    # Fall back to dynamic shape if static shape is not available
                    num_channels = tf.shape(y_pred)[-1]
            except:
                # Ultimate fallback - assume a default value
                num_channels = args.fse_channels
            
            # Use tf.range instead of Python's range for tensor-compatible iteration
            loss = tf.constant(0.0, dtype=tf.float32)
            
            # Use tf.function compatible approach to iterate over channels
            def body(i, current_loss):
                channel = tf.gather(y_pred, i, axis=-1)
                flat_channel = tf.reshape(channel, [-1])
                
                # Create histogram
                hist = tf.histogram_fixed_width(flat_channel, [0.0, 1.0], nbins=nbins)
                hist = tf.cast(hist, tf.float32) / tf.cast(tf.size(flat_channel), tf.float32)
                hist = hist + epsilon
                
                # Calculate KL divergence to uniform distribution
                kl_div = tf.reduce_sum(hist * tf.math.log(hist / target_dist))
                return i + 1, current_loss + kl_div
            
            # Only use while_loop for dynamic shapes, use unrolled loop for static shapes
            if isinstance(num_channels, tf.Tensor):
                # Use while_loop for dynamic shape
                _, total_loss = tf.while_loop(
                    lambda i, _: i < num_channels,
                    body,
                    [tf.constant(0), loss]
                )
                loss = total_loss / tf.cast(num_channels, tf.float32)
            else:
                # Unroll the loop for static shape (more efficient)
                total_loss = 0.0
                for i in range(num_channels):
                    channel = y_pred[..., i]
                    flat_channel = tf.reshape(channel, [-1])
                    
                    # Create histogram
                    hist = tf.histogram_fixed_width(flat_channel, [0.0, 1.0], nbins=nbins)
                    hist = tf.cast(hist, tf.float32) / tf.cast(tf.size(flat_channel), tf.float32)
                    hist = hist + epsilon
                    
                    # Calculate KL divergence to uniform distribution
                    kl_div = tf.reduce_sum(hist * tf.math.log(hist / target_dist))
                    total_loss += kl_div
                
                loss = total_loss / num_channels
            
            return loss
        
        # === Enhanced color fidelity loss with distribution regularization ===
        def improved_color_fidelity_loss(y_true, y_pred):
            """
            Enhanced loss function focusing on color fidelity and float distribution.
            Note: y_true is the original image, y_pred is the expanded channel encoding.
            We'll use only first 3 channels for comparison with original image.
            """
            # For image comparison, use only first 3 channels of encoding
            # Handle dynamic tensor shapes properly
            y_pred_rgb = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, 3])
            
            # Basic MSE for pixel-level accuracy
            mse = tf.reduce_mean(tf.square(y_true - y_pred_rgb))
            
            # PSNR-based component to encourage high reconstruction quality
            psnr = tf.image.psnr(y_true, y_pred_rgb, max_val=1.0)
            psnr_loss = tf.reduce_mean(tf.maximum(0.0, 30.0 - psnr)) / 30.0
            
            # Color space specific component
            y_true_yuv = tf.image.rgb_to_yuv(y_true)
            y_pred_yuv = tf.image.rgb_to_yuv(y_pred_rgb)
            
            # Split channels - using tf.split instead of unpacking to avoid tensor shape issues
            y_true_channels = tf.split(y_true_yuv, num_or_size_splits=3, axis=-1)
            y_pred_channels = tf.split(y_pred_yuv, num_or_size_splits=3, axis=-1)
            
            y_true_y, y_true_u, y_true_v = y_true_channels[0], y_true_channels[1], y_true_channels[2]
            y_pred_y, y_pred_u, y_pred_v = y_pred_channels[0], y_pred_channels[1], y_pred_channels[2]
            
            # Compute luminance error (Y channel)
            luminance_error = tf.reduce_mean(tf.square(y_true_y - y_pred_y))
            
            # Compute chrominance error (U & V channels)
            chrominance_error = tf.reduce_mean(tf.square(y_true_u - y_pred_u) + 
                                            tf.square(y_true_v - y_pred_v))
            
            # Add distribution regularization for float space - using a separate function
            # to avoid the SymbolicTensor issue
            dist_loss = distribution_regularization(y_pred)
            
            # Get the weight as a constant to avoid any potential graph issues
            distribution_weight = tf.constant(args.distribution_weight, dtype=tf.float32)
            
            # Combined loss with emphasis on color fidelity
            return 0.5 * mse + 0.1 * psnr_loss + 0.1 * luminance_error + 0.2 * chrominance_error + distribution_weight * dist_loss

        def distribution_regularization(y_pred):
            """Separate function for distribution regularization to avoid tensor issues"""
            # Add small epsilon to avoid log(0)
            epsilon = 1e-7
            
            # Calculate histogram with fixed bins for stability
            nbins = 20
            target_dist = tf.ones([nbins]) / nbins  # Uniform target distribution
            
            # Get the static number of channels if possible
            num_channels = y_pred.shape[-1]
            if num_channels is None:
                # If shape is dynamic, use a safe default
                num_channels = args.fse_channels
            
            # Calculate distribution loss for each channel
            channel_losses = []
            
            # Safe loop for static shapes
            for i in range(num_channels):
                # Extract single channel values
                channel = tf.slice(y_pred, [0, 0, 0, i], [-1, -1, -1, 1])
                channel = tf.squeeze(channel, axis=-1)  # Remove the channel dimension
                flat_channel = tf.reshape(channel, [-1])
                
                # Create histogram
                hist = tf.histogram_fixed_width(flat_channel, [0.0, 1.0], nbins=nbins)
                hist = tf.cast(hist, tf.float32) / tf.cast(tf.size(flat_channel), tf.float32)
                hist = hist + epsilon
                
                # Calculate KL divergence to uniform distribution
                kl_div = tf.reduce_sum(hist * tf.math.log(hist / target_dist))
                channel_losses.append(kl_div)
            
            # Average all channel losses
            total_loss = tf.math.add_n(channel_losses)
            avg_loss = total_loss / tf.cast(len(channel_losses), tf.float32)
            
            return avg_loss
        
        # === Decoder supervision loss ===# === Decoder supervision loss ===
        def decoder_rgb_loss(y_true, y_pred):
            """Loss for the decoded RGB image to ensure encodings are decodable."""
            # MSE for basic pixel accuracy
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # SSIM for structural similarity
            ssim = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
            
            # PSNR component to encourage high reconstruction quality
            psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
            psnr_loss = tf.reduce_mean(tf.maximum(0.0, 30.0 - psnr)) / 30.0
            
            # Color space specific component for color accuracy
            y_true_yuv = tf.image.rgb_to_yuv(y_true)
            y_pred_yuv = tf.image.rgb_to_yuv(y_pred)
            
            # Split channels - using tf.split instead of unpacking to avoid tensor shape issues
            y_true_channels = tf.split(y_true_yuv, num_or_size_splits=3, axis=-1)
            y_pred_channels = tf.split(y_pred_yuv, num_or_size_splits=3, axis=-1)
            
            y_true_y, y_true_u, y_true_v = y_true_channels[0], y_true_channels[1], y_true_channels[2]
            y_pred_y, y_pred_u, y_pred_v = y_pred_channels[0], y_pred_channels[1], y_pred_channels[2]
            
            # Compute color errors
            luminance_error = tf.reduce_mean(tf.square(y_true_y - y_pred_y))
            chrominance_error = tf.reduce_mean(tf.square(y_true_u - y_pred_u) + 
                                            tf.square(y_true_v - y_pred_v))
            
            # Combined decoder loss with emphasis on accurate reconstruction
            return 0.4 * mse + 0.2 * ssim + 0.1 * psnr_loss + 0.1 * luminance_error + 0.2 * chrominance_error

        # === Metrics ===
        def mae_sliced_metric(y_true, y_pred):
            """MAE that compares only the first 3 channels of predicted tensor."""
            y_pred_rgb = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, 3])
            return tf.reduce_mean(tf.abs(y_true - y_pred_rgb))

        def psnr_metric(y_true, y_pred):
            """Peak Signal-to-Noise Ratio metric for image quality."""
            y_pred_rgb = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, 3])
            return tf.image.psnr(y_true, y_pred_rgb, max_val=1.0)

        def ssim_metric(y_true, y_pred):
            """Structural Similarity Index metric for perceptual quality."""
            y_pred_rgb = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, 3])
            return tf.image.ssim(y_true, y_pred_rgb, max_val=1.0)

        def yuv_psnr_metric(y_true, y_pred):
            """PSNR in YUV colorspace to measure color fidelity."""
            y_pred_rgb = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, 3])
            y_true_yuv = tf.image.rgb_to_yuv(y_true)
            y_pred_yuv = tf.image.rgb_to_yuv(y_pred_rgb)
            return tf.image.psnr(y_true_yuv, y_pred_yuv, max_val=1.0)

        
        def float_distribution_metric(y_true, y_pred):
            """Track how well-distributed the float values are."""
            return float_distribution_loss(y_true, y_pred)

        # === Assign losses ===
        losses = {
            "lithos_base_encoding": improved_color_fidelity_loss,
            "lithos_material_class": "categorical_crossentropy",
            "lithos_material_properties": "mse"
        }
        
        # Add decoder loss if in training mode and using supervision
        if self.use_decoder_supervision and "decoded_rgb" in self.model.output_names:
            losses["decoded_rgb"] = decoder_rgb_loss

        if self.use_importance_map:
            losses["channel_importance"] = "binary_crossentropy"

        # === Assign metrics ===
        metrics = {
            "lithos_base_encoding": [mae_sliced_metric, psnr_metric, ssim_metric, yuv_psnr_metric, float_distribution_metric],
            "lithos_material_class": ["accuracy"],
            "lithos_material_properties": ["mae"]
        }
        
        # Add metrics for decoder output
        if self.use_decoder_supervision and "decoded_rgb" in self.model.output_names:
            metrics["decoded_rgb"] = ["mae", psnr_metric, ssim_metric, yuv_psnr_metric]

        if self.use_importance_map:
            metrics["channel_importance"] = ["mae"]

        # === Compile ===
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=losses,
            loss_weights=self.loss_weights,
            metrics=metrics
        )

    def get_summary(self):
        """Return model summary as string."""
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + "\n"))
        return stream.getvalue()

    def _inverted_res_block(self, x, filters, strides, expansion, block_id):
        """Custom MobileNet-style inverted residual block."""
        prefix = f'block_{block_id}_'

        shortcut = x
        in_channels = x.shape[-1]

        expanded = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                          use_bias=False, name=prefix + 'expand')(x)
        expanded = BatchNormalization(name=prefix + 'expand_bn')(expanded)
        expanded = Activation('relu')(expanded)

        depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same',
                                                    use_bias=False, name=prefix + 'depthwise')(expanded)
        depthwise = BatchNormalization(name=prefix + 'depthwise_bn')(depthwise)
        depthwise = Activation('relu')(depthwise)

        projected = Conv2D(filters, kernel_size=1, padding='same',
                           use_bias=False, name=prefix + 'project')(depthwise)
        projected = BatchNormalization(name=prefix + 'project_bn')(projected)

        if strides == 1 and in_channels == filters:
            return tf.keras.layers.Add(name=prefix + 'add')([shortcut, projected])
        else:
            return projected
            
    def load_checkpoint_with_channel_expansion(self, checkpoint_path):
        """
        Load checkpoint while handling the architecture changes:
        1. Expanding base_encoding from 3 to 6 channels
        2. Adding decoder branch
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            tuple: (success, epoch, step)
        """
        try:
            # Extract epoch and step from filename
            filename = os.path.basename(checkpoint_path)
            parts = filename.split('_')
            epoch = int(parts[-2].replace('ep', '')) if len(parts) > 2 else 0
            step = int(parts[-1].split('.')[0].replace('step', '')) if len(parts) > 1 else 0
            
            # Ensure proper GCS path format
            if checkpoint_path.startswith('gs://'):
                # Full GCS path provided
                gcs_path = checkpoint_path
                # Extract the part after bucket name
                path_without_bucket = '/'.join(checkpoint_path.split('/')[3:])
            else:
                # Relative path provided - construct full GCS path
                gcs_path = f"gs://{GCS_BUCKET_NAME}/{checkpoint_path}"
                path_without_bucket = checkpoint_path
            
            logger.info(f"🔍 Loading weights from {gcs_path}")
            
            # Prepare local directory
            local_dir = os.path.join(LOCAL_TEMP_DIR, 'checkpoints')
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            
            # Download the checkpoint file from GCS
            try:
                logger.info(f"📥 Downloading checkpoint: {path_without_bucket}")
                blob = bucket.blob(path_without_bucket)
                if not blob.exists():
                    logger.error(f"❌ Checkpoint file not found in GCS: {path_without_bucket}")
                    # Try direct path without bucket prefix
                    direct_path = checkpoint_path.replace(f"gs://{GCS_BUCKET_NAME}/", "")
                    logger.info(f"🔄 Trying alternative path: {direct_path}")
                    blob = bucket.blob(direct_path)
                    if not blob.exists():
                        raise FileNotFoundError(f"Checkpoint file not found in GCS: {direct_path}")
                    else:
                        logger.info(f"✅ Found checkpoint using alternative path")
                
                blob.download_to_filename(local_path)
                logger.info(f"✅ Downloaded checkpoint to {local_path}")
                
                # Verify file was downloaded correctly
                if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                    raise FileNotFoundError(f"Downloaded file is empty or missing: {local_path}")
                
            except Exception as download_error:
                logger.error(f"❌ Error downloading checkpoint: {download_error}")
                raise
            
            # Create temporary storage for weights
            weights_dict = {}
            
            # Try direct loading first (will likely fail for architecture changes)
            try:
                load_status = self.model.load_weights(local_path).expect_partial()
                logger.info("✅ Weights loaded successfully")
                return True, epoch, step
            except Exception as e:
                logger.warning(f"⚠️ Complete loading failed as expected: {e}")
                logger.info("🔄 Attempting layer-by-layer loading with shape adaptation...")
            
            # Load weights from the checkpoint file using h5py for manual loading
            try:
                import h5py
                with h5py.File(local_path, 'r') as f:
                    # Get weight names and shapes
                    for key in f.keys():
                        group = f[key]
                        if 'layer_names' in group.attrs:
                            layer_names = [n.decode('utf8') for n in group.attrs['layer_names']]
                            for name in layer_names:
                                layer_group = group[name]
                                if 'weight_names' in layer_group.attrs:
                                    weight_names = [n.decode('utf8') for n in layer_group.attrs['weight_names']]
                                    weights = [np.array(layer_group[weight_name]) for weight_name in weight_names]
                                    if weights:
                                        weights_dict[name] = weights
            except Exception as h5py_error:
                logger.error(f"❌ Error reading H5 file: {h5py_error}")
                raise
            
            # Load weights for all layers except the final lithos_base_encoding layer
            for layer in self.model.layers:
                if layer.name in weights_dict and layer.name != 'lithos_base_encoding':
                    try:
                        layer.set_weights(weights_dict[layer.name])
                        logger.info(f"✓ Loaded weights for {layer.name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load weights for {layer.name}: {e}")
            
            # Special handling for lithos_base_encoding layer
            if 'lithos_base_encoding' in weights_dict:
                try:
                    old_weights = weights_dict['lithos_base_encoding']
                    new_layer = self.model.get_layer('lithos_base_encoding')
                    new_weights = new_layer.get_weights()
                    
                    # Check shapes
                    old_shape = old_weights[0].shape
                    new_shape = new_weights[0].shape
                    
                    logger.info(f"Old base_encoding shape: {old_shape}, New shape: {new_shape}")
                    
                    # Copy weights for the first 3 channels, initialize the rest
                    if old_shape[-1] == 3 and new_shape[-1] > 3:
                        # Copy kernel weights
                        new_weights[0][..., :3] = old_weights[0]
                        
                        # Copy bias if exists
                        if len(old_weights) > 1 and len(new_weights) > 1:
                            new_weights[1][:3] = old_weights[1]
                        
                        # Set weights with combined old and new
                        new_layer.set_weights(new_weights)
                        logger.info(f"✅ Successfully expanded base_encoding from 3 to {new_shape[-1]} channels")
                    else:
                        logger.warning(f"⚠️ Unexpected shape change in base_encoding: {old_shape} -> {new_shape}")
                        logger.warning("⚠️ Initializing base_encoding with random weights")
                except Exception as e:
                    logger.error(f"❌ Failed to adapt base_encoding layer: {e}")
            
            logger.info("✅ Checkpoint loaded with architecture adaptation")
            return True, epoch, step
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, 0, 0

class MultiViewLithosModel(LithosModel):
    """Extended LITHOS model that incorporates view angle conditioning."""

    def __init__(self,
                 input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
                 backbone_type="mobilenet_v2",
                 training=False,
                 learning_rate=0.001,
                 dropout_rate=0.1,
                 l2_reg=1e-6,
                 loss_weights=None,
                 num_view_angles=8,
                 debug_mode=False,
                 use_importance_map=False,
                 fse_channels=6,
                 use_decoder_supervision=False):
                 
        self.num_view_angles = num_view_angles
        self.debugMode = debug_mode
        super().__init__(
            input_shape=input_shape,
            backbone_type=backbone_type,
            training=training,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            loss_weights=loss_weights,
            use_importance_map=use_importance_map,
            fse_channels=fse_channels,
            use_decoder_supervision=use_decoder_supervision
        )

    def _build_model(self):
        """Build multiview model with angle input."""
        image_input = Input(shape=self.input_shape, name="image_input")
        angle_input = Input(shape=(self.num_view_angles,), name="view_angle_input")

        backbone_output = self._build_backbone(image_input)
        conditioned = self._condition_on_view_angle(backbone_output, angle_input)

        base_encoding = self._build_base_encoding_branch(conditioned)
        material_class = self._build_material_class_branch(conditioned)
        material_properties = self._build_material_properties_branch(conditioned)
        
        outputs = {
            "lithos_base_encoding": base_encoding,
            "lithos_material_class": material_class,
            "lithos_material_properties": material_properties
        }
        
        # Add decoder branch for supervision
        if self.training and self.use_decoder_supervision:
            decoded_rgb = self._build_decoder_branch(base_encoding)
            outputs["decoded_rgb"] = decoded_rgb
        
        # Add importance map branch if enabled
        if self.use_importance_map:
            importance_map = self._build_importance_map_branch(conditioned)
            outputs["channel_importance"] = importance_map

        self.model = Model(
            inputs=[image_input, angle_input],
            outputs=outputs,
            name="lithos_multiview_model"
        )

    def _condition_on_view_angle(self, features, view_angle_encoding):
        """Merge view angle encoding into spatial features (broadcast + conv)."""
        view_feat = Dense(32, activation='relu', name='view_angle_dense1')(view_angle_encoding)
        view_feat = Dense(64, activation='relu', name='view_angle_dense2')(view_feat)

        view_feat_exp = tf.expand_dims(tf.expand_dims(view_feat, 1), 1)
        view_feat_tiled = tf.tile(view_feat_exp, [1, tf.shape(features)[1], tf.shape(features)[2], 1])

        combined = Concatenate(axis=-1)([features, view_feat_tiled])

        x = Conv2D(features.shape[-1], 1, padding='same',
                   kernel_regularizer=regularizers.l2(self.l2_reg),
                   name='view_conditioning_conv')(combined)
        x = BatchNormalization(name='view_conditioning_bn')(x)
        x = Activation('relu', name='view_conditioning_relu')(x)

        return x

# Function to enhance LITHOS model with importance-based quantization
def enhance_lithos_quantization(model):
    """
    Add a learnable importance predictor branch to an existing LITHOS model.
    
    Args:
        model: An instance of LithosModel or MultiViewLithosModel
        
    Returns:
        Enhanced model with quantization capabilities
    """
    # Create importance map branch
    if isinstance(model, MultiViewLithosModel):
        # For multi-view model, we need a new instance with importance map
        enhanced_model = MultiViewLithosModel(
            input_shape=model.input_shape,
            backbone_type=model.backbone_type,
            training=model.training,
            learning_rate=model.learning_rate,
            dropout_rate=model.dropout_rate,
            l2_reg=model.l2_reg,
            loss_weights=model.loss_weights,
            num_view_angles=model.num_view_angles,
            debug_mode=model.debugMode,
            use_importance_map=True, 
            fse_channels=model.fse_channels,
            use_decoder_supervision=model.use_decoder_supervision
        )
    else:
        # For standard model
        enhanced_model = LithosModel(
            input_shape=model.input_shape,
            backbone_type=model.backbone_type,
            training=model.training,
            learning_rate=model.learning_rate,
            dropout_rate=model.dropout_rate,
            l2_reg=model.l2_reg,
            loss_weights=model.loss_weights,
            use_importance_map=True,
            fse_channels=model.fse_channels,
            use_decoder_supervision=model.use_decoder_supervision
        )
    
    # Copy weights from original model to enhanced model for common layers
    # Get all layers from original model
    for layer in model.model.layers:
        # Try to find the same layer in the enhanced model
        try:
            if layer.name in [l.name for l in enhanced_model.model.layers]:
                enhanced_model.model.get_layer(layer.name).set_weights(layer.get_weights())
        except:
            continue
    
    logger.info("Enhanced LITHOS model with importance-based quantization")
    return enhanced_model

# === Checkpoint Utilities ===

def save_checkpoint_to_gcs(model, step, epoch, optimizer_weights=None):
    """Save model weights and optional optimizer state to GCS."""
    try:
        ckpt_dir = os.path.join(LOCAL_TEMP_DIR, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)

        local_path = os.path.join(ckpt_dir, f'lithos_checkpoint_ep{epoch}_step{step}.weights.h5')
        model.model.save_weights(local_path)
        logger.info(f"📦 Saved checkpoint locally: {local_path}")

        blob_path = f"{GCS_CHECKPOINT_DIR}/lithos_checkpoint_ep{epoch}_step{step}.weights.h5"
        bucket.blob(blob_path).upload_from_filename(local_path)

        if optimizer_weights is not None:
            opt_path = os.path.join(ckpt_dir, f'optimizer_ep{epoch}_step{step}.npy')
            np.save(opt_path, optimizer_weights)
            bucket.blob(f"{GCS_CHECKPOINT_DIR}/optimizer_ep{epoch}_step{step}.npy").upload_from_filename(opt_path)

        metadata = {
            "epoch": epoch,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "input_size": INPUT_SIZE,
            "multi_view": args.multi_view,
            "backbone": args.backbone,
            "importance_map": args.use_importance_map,
            "enhanced_augmentation": args.use_enhanced_augmentation,
            "fse_channels": args.fse_channels,
            "use_decoder_supervision": args.use_decoder_supervision
        }
        meta_path = os.path.join(ckpt_dir, f'metadata_ep{epoch}_step{step}.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
        bucket.blob(f"{GCS_CHECKPOINT_DIR}/metadata_ep{epoch}_step{step}.json").upload_from_filename(meta_path)

        logger.info(f"✅ Checkpoint uploaded to GCS at step {step}, epoch {epoch}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to save checkpoint: {e}")
        return False

def find_latest_checkpoint_in_gcs(target_step=None):
    """
    Get checkpoint filename, epoch, and step from GCS.
    Can find latest checkpoint or specific step if provided.
    
    Args:
        target_step: If provided, find checkpoint at this specific step
                     
    Returns:
        tuple: (checkpoint_path, epoch, step)
    """
    try:
        logger.info(f"🔍 Searching for checkpoints in {GCS_CHECKPOINT_DIR}...")
        
        # List all blobs in the checkpoint directory
        blobs = list(bucket.list_blobs(prefix=f"{GCS_CHECKPOINT_DIR}/"))
        
        # Log what we found for debugging
        logger.info(f"Found {len(blobs)} blobs with prefix {GCS_CHECKPOINT_DIR}/")
        
        # Filter for checkpoint files with the right pattern
        checkpoints = []
        for b in blobs:
            if 'lithos_checkpoint_ep' in b.name and b.name.endswith('.weights.h5'):
                try:
                    name = os.path.basename(b.name)
                    
                    # Extract epoch and step from filename
                    # Expected format: lithos_checkpoint_ep{EPOCH}_step{STEP}.weights.h5
                    if 'ep' in name and 'step' in name:
                        parts = name.split('_')
                        epoch_part = [p for p in parts if p.startswith('ep')][0]
                        step_part = [p for p in parts if p.startswith('step')][0]
                        
                        epoch = int(epoch_part.replace('ep', ''))
                        step = int(step_part.split('.')[0].replace('step', ''))
                        
                        checkpoints.append((b.name, epoch, step, b.updated))
                        logger.info(f"  Found checkpoint: {b.name}, epoch {epoch}, step {step}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not parse checkpoint name: {b.name}, error: {e}")
                    continue
        
        if not checkpoints:
            logger.info("⚠️ No checkpoints found in bucket.")
            return None, None, None
            
        # If a specific step is requested, find it
        if target_step is not None:
            logger.info(f"🎯 Looking for checkpoint at step {target_step}")
            
            # First try exact match
            exact_matches = [c for c in checkpoints if c[2] == target_step]
            if exact_matches:
                match = max(exact_matches, key=lambda x: x[1])  # Get highest epoch for this step
                logger.info(f"✅ Found exact step match: {os.path.basename(match[0])}")
                full_path = f"gs://{GCS_BUCKET_NAME}/{match[0]}"
                return full_path, match[1], match[2]
                
            # Try finding closest step
            logger.info(f"⚠️ No exact match for step {target_step}, looking for closest step")
            closest_step = min(checkpoints, key=lambda x: abs(x[2] - target_step))
            logger.info(f"📍 Found closest step: {closest_step[2]} in {os.path.basename(closest_step[0])}")
            full_path = f"gs://{GCS_BUCKET_NAME}/{closest_step[0]}"
            return full_path, closest_step[1], closest_step[2]
            
        # Otherwise get latest checkpoint (first by epoch, then by step)
        logger.info("🔍 Finding latest checkpoint by epoch and step")
        latest = max(checkpoints, key=lambda x: (x[1], x[2]))
        logger.info(f"📍 Latest checkpoint: Epoch {latest[1]}, Step {latest[2]} in {os.path.basename(latest[0])}")
        full_path = f"gs://{GCS_BUCKET_NAME}/{latest[0]}"
        return full_path, latest[1], latest[2]

    except Exception as e:
        logger.error(f"❌ Error finding checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

def load_checkpoint_with_fallback(model, target_step=None):
    """
    Load checkpoint with better error handling and fallback options.
    
    Args:
        model: The model to load weights into
        target_step: Specific step to load (if None, loads latest)
        
    Returns:
        tuple: (success, epoch, step)
    """
    # If architecture has changed and fse_channels is greater than 3 or decoder supervision is enabled,
    # use specialized loading method
    if args.fse_channels > 3 or args.use_decoder_supervision:
        if args.checkpoint_path:
            return model.load_checkpoint_with_channel_expansion(args.checkpoint_path)
        
        checkpoint_path, epoch, step = find_latest_checkpoint_in_gcs(target_step)
        if checkpoint_path:
            return model.load_checkpoint_with_channel_expansion(checkpoint_path)
        return False, 0, 0
    
    # Otherwise use standard loading
    checkpoint_path, epoch, step = find_latest_checkpoint_in_gcs(target_step)
    
    if checkpoint_path is None:
        logger.warning("⚠️ No checkpoint found to load")
        return False, 0, 0
    
    ckpt_dir = os.path.join(LOCAL_TEMP_DIR, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    local_path = os.path.join(ckpt_dir, os.path.basename(checkpoint_path))
    
    try:
        # Download the checkpoint file
        logger.info(f"📥 Downloading checkpoint: {checkpoint_path}")
        bucket.blob(checkpoint_path).download_to_filename(local_path)
        
        # Check if file exists and has content
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            logger.error(f"❌ Downloaded checkpoint file is empty or missing: {local_path}")
            return False, 0, 0
            
        # Try loading weights
        logger.info(f"🔄 Loading weights from {local_path}")
        
        # For multi-view model, we need to prepare the model before loading
        is_multiview = isinstance(model, MultiViewLithosModel)
        if is_multiview:
            # Create a batch of dummy inputs to build model
            dummy_img = tf.zeros((1, INPUT_SIZE, INPUT_SIZE, 3))
            dummy_angle = tf.zeros((1, args.num_view_angles))
            # Call the model once to build it
            _ = model.model([dummy_img, dummy_angle])
            
        # Now load weights
        model.model.load_weights(local_path)
        logger.info(f"✅ Successfully loaded checkpoint from epoch {epoch}, step {step}")
        return True, epoch, step
        
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # If the first attempt failed, try loading an earlier checkpoint
        try:
            if target_step is not None:
                logger.info(f"🔄 First attempt failed, trying to find an earlier checkpoint")
                prev_checkpoints = list(bucket.list_blobs(prefix=f"{GCS_CHECKPOINT_DIR}/"))
                
                valid_checkpoints = []
                for b in prev_checkpoints:
                    if 'lithos_checkpoint_ep' in b.name and b.name.endswith('.weights.h5'):
                        try:
                            name = os.path.basename(b.name)
                            if 'ep' in name and 'step' in name:
                                parts = name.split('_')
                                epoch_part = [p for p in parts if p.startswith('ep')][0]
                                step_part = [p for p in parts if p.startswith('step')][0]
                                
                                e = int(epoch_part.replace('ep', ''))
                                s = int(step_part.split('.')[0].replace('step', ''))
                                
                                if s < step:  # Earlier step
                                    valid_checkpoints.append((b.name, e, s))
                        except:
                            continue
                
                if valid_checkpoints:
                    # Get the most recent earlier checkpoint
                    latest_prev = max(valid_checkpoints, key=lambda x: (x[1], x[2]))
                    logger.info(f"🔍 Trying earlier checkpoint: Epoch {latest_prev[1]}, Step {latest_prev[2]}")
                    
                    alt_path = os.path.join(ckpt_dir, os.path.basename(latest_prev[0]))
                    bucket.blob(latest_prev[0]).download_to_filename(alt_path)
                    
                    model.model.load_weights(alt_path)
                    logger.info(f"✅ Successfully loaded earlier checkpoint")
                    return True, latest_prev[1], latest_prev[2]
        except Exception as fallback_error:
            logger.error(f"❌ Fallback loading also failed: {fallback_error}")
        
        return False, 0, 0

def load_checkpoint_from_gcs(model, checkpoint_path=None):
    """Download and load model weights from GCS."""
    if checkpoint_path is None:
        checkpoint_path, epoch, step = find_latest_checkpoint_in_gcs()
        if checkpoint_path is None:
            return False, 0, 0
    else:
        name = os.path.basename(checkpoint_path)
        epoch = int(name.split('_')[-2].replace('ep', ''))
        step = int(name.split('_')[-1].replace('step', ''))

    try:
        ckpt_dir = os.path.join(LOCAL_TEMP_DIR, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        local_path = os.path.join(ckpt_dir, os.path.basename(checkpoint_path))
        bucket.blob(checkpoint_path).download_to_filename(local_path)

        model.model.load_weights(local_path)
        logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")
        return True, epoch, step
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        return False, 0, 0

def save_final_model_to_gcs(model, epoch, step, metrics=None):
    """Export final model and training metrics to GCS."""
    try:
        export_dir = os.path.join(LOCAL_TEMP_DIR, 'final_model')
        os.makedirs(export_dir, exist_ok=True)

        model_path = os.path.join(export_dir, 'lithos_model.h5')
        tflite_path = os.path.join(export_dir, 'lithos_model.tflite')
        summary_path = os.path.join(export_dir, 'model_summary.txt')

        model.model.save(model_path)

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        # Save model summary
        with open(summary_path, 'w') as f:
            f.write(model.get_summary())

        if metrics:
            metrics_path = os.path.join(export_dir, 'training_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

        # Upload all files to GCS
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_type = []
        if args.multi_view:
            model_type.append("multiview")
        if args.use_importance_map:
            model_type.append("importance")
        if args.use_enhanced_augmentation:
            model_type.append("enhanced-aug")
        if args.fse_channels > 3:
            model_type.append(f"fse{args.fse_channels}")
        if args.use_decoder_supervision:
            model_type.append("decoder")
            
        model_type_str = "_".join(model_type) if model_type else "standard"
        gcs_prefix = f"{GCS_BASE_PATH}/final_models/lithos_{model_type_str}_{timestamp}_ep{epoch}_step{step}"

        for local_file in ['lithos_model.h5', 'lithos_model.tflite', 'model_summary.txt']:
            blob = bucket.blob(f"{gcs_prefix}/{local_file}")
            blob.upload_from_filename(os.path.join(export_dir, local_file))

        if metrics:
            blob = bucket.blob(f"{gcs_prefix}/training_metrics.json")
            blob.upload_from_filename(metrics_path)

        logger.info(f"✅ Final model saved and uploaded to {gcs_prefix}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to save final model: {e}")
        return False

def train_lithos_model():
    """Train the LITHOS model with enhanced architecture."""
    logger.info("🚀 Starting LITHOS training with enhanced architecture...")

    # GPU memory growth
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"✅ Enabled GPU memory growth for {len(gpus)} GPU(s)")
    except Exception as e:
        logger.warning(f"⚠️ Failed to configure GPU memory: {e}")

    all_samples = find_common_samples(training=True)
    if args.max_samples:
        all_samples = all_samples[:args.max_samples]

    if args.use_validation:
        train_samples, val_samples = split_train_val_data(all_samples)
    else:
        train_samples = all_samples
        val_samples = None
    
    if args.max_val_samples and val_samples:
        val_samples = val_samples[:args.max_val_samples]

    if not train_samples:
        logger.error("❌ No training data found.")
        return False

    logger.info(f"🧪 Training samples: {len(train_samples)}")
    if val_samples:
        logger.info(f"🧪 Validation samples: {len(val_samples)}")

    train_dataset = create_lithos_dataset(train_samples, args.batch_size, training=True)
    val_dataset = create_lithos_dataset(val_samples, args.val_batch_size, training=False) if val_samples else None

    steps_per_epoch = max(1, len(train_samples) // args.batch_size)
    validation_steps = max(1, len(val_samples) // args.val_batch_size) if val_samples else None

    # Initialize the model with enhanced architecture options
    model = LithosModel(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        backbone_type=args.backbone,
        training=True,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg,
        loss_weights={
            "lithos_base_encoding": args.base_encoding_weight,
            "lithos_material_class": args.material_class_weight,
            "lithos_material_properties": 0.3,
            "decoded_rgb": args.decoded_rgb_weight if args.use_decoder_supervision else 0.0,
        },
        use_importance_map=args.use_importance_map,
        fse_channels=args.fse_channels,
        use_decoder_supervision=args.use_decoder_supervision
    )

    logger.info("\n" + model.get_summary())

    initial_epoch = args.initial_epoch
    initial_step = 0
    if not args.skip_checkpoint:
        # Use improved checkpoint loading with special handling for architecture changes
        if args.fse_channels > 3 or args.use_decoder_supervision:
            # Use special loading method for architecture changes
            if args.checkpoint_path:
                success, initial_epoch, initial_step = model.load_checkpoint_with_channel_expansion(args.checkpoint_path)
            else:
                success, initial_epoch, initial_step = load_checkpoint_with_fallback(model, args.load_step)
        else:
            # Standard loading for unchanged architecture
            success, initial_epoch, initial_step = load_checkpoint_from_gcs(model)

    class CheckpointCallback(tf.keras.callbacks.Callback):
        def __init__(self, step, epoch):
            self.step = step
            self.epoch = epoch

        def on_batch_end(self, batch, logs=None):
            self.step += 1
            if self.step % args.checkpoint_steps == 0:
                save_checkpoint_to_gcs(model, self.step, self.epoch)

        def on_epoch_end(self, epoch, logs=None):
            self.epoch = initial_epoch + epoch + 1
            save_checkpoint_to_gcs(model, self.step, self.epoch)
            
    callbacks = [CheckpointCallback(initial_step, initial_epoch)]

    if val_dataset and args.early_stopping_patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.early_stopping_patience,
            restore_best_weights=True, verbose=1
        ))

    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss' if val_dataset else 'loss',
        factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ))

    logger.info(f"📊 Training with {args.fse_channels} FSE channels and {'with' if args.use_decoder_supervision else 'without'} decoder supervision")
    logger.info(f"📊 Training from epoch {initial_epoch} to {initial_epoch + args.epochs}")
    history = model.model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=initial_epoch,
        epochs=initial_epoch + args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_path = os.path.join(LOCAL_TEMP_DIR, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    # Add architecture info to history filename
    arch_info = f"fse{args.fse_channels}"
    if args.use_decoder_supervision:
        arch_info += "_decoder"
    
    hist_blob = bucket.blob(f"{GCS_BASE_PATH}/training_history/history_standard_{arch_info}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    hist_blob.upload_from_filename(hist_path)

    final_epoch = initial_epoch + args.epochs - 1
    final_step = callbacks[0].step
    save_final_model_to_gcs(model, final_epoch, final_step, history_dict)

    logger.info("✅ Enhanced LITHOS training completed!")
    return True
    
def train_multi_view_lithos_model():
    """Train multi-view LITHOS model with enhanced architecture options."""
    logger.info("🚀 Starting LITHOS multi-view training with enhanced architecture...")

    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"✅ Enabled memory growth for {len(gpus)} GPU(s)")
    except Exception as e:
        logger.warning(f"⚠️ Could not configure GPU: {e}")

    all_products = find_multi_view_products(training=True)
    if args.max_samples:
        all_products = all_products[:args.max_samples]

    val_products = find_multi_view_products(training=False) if args.use_validation else None
    if args.max_val_samples and val_products:
        val_products = val_products[:args.max_val_samples]

    if not all_products:
        logger.error("❌ No multi-view products found.")
        return False

    train_dataset = create_multi_view_lithos_dataset(all_products, args.batch_size, training=True)
    val_dataset = create_multi_view_lithos_dataset(val_products, args.val_batch_size, training=False) if val_products else None

    steps_per_epoch = max(1, len(all_products) // args.batch_size)
    validation_steps = max(1, len(val_products) // args.val_batch_size) if val_products else None

    model = MultiViewLithosModel(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        backbone_type=args.backbone,
        training=True,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg,
        loss_weights={
            "lithos_base_encoding": args.base_encoding_weight,
            "lithos_material_class": args.material_class_weight,
            "lithos_material_properties": 0.3,
            "decoded_rgb": args.decoded_rgb_weight if args.use_decoder_supervision else 0.0
        },
        num_view_angles=args.num_view_angles,
        use_importance_map=args.use_importance_map,
        fse_channels=args.fse_channels,
        use_decoder_supervision=args.use_decoder_supervision
    )

    logger.info("\n" + model.get_summary())

    initial_epoch = args.initial_epoch
    initial_step = 0
    if not args.skip_checkpoint:
        # Use improved checkpoint loading with special handling for architecture changes
        if args.fse_channels > 3 or args.use_decoder_supervision:
            # Use special loading method for architecture changes
            if args.checkpoint_path:
                success, initial_epoch, initial_step = model.load_checkpoint_with_channel_expansion(args.checkpoint_path)
            else:
                success, initial_epoch, initial_step = load_checkpoint_with_fallback(model, args.load_step)
        else:
            # Standard loading for unchanged architecture
            success, initial_epoch, initial_step = load_checkpoint_from_gcs(model)

    class CheckpointCallback(tf.keras.callbacks.Callback):
        def __init__(self, step, epoch):
            self.step = step
            self.epoch = epoch

        def on_batch_end(self, batch, logs=None):
            self.step += 1
            if self.step % args.checkpoint_steps == 0:
                save_checkpoint_to_gcs(model, self.step, self.epoch)

        def on_epoch_end(self, epoch, logs=None):
            self.epoch = initial_epoch + epoch + 1
            save_checkpoint_to_gcs(model, self.step, self.epoch)
    callbacks = [CheckpointCallback(initial_step, initial_epoch)]

    if val_dataset and args.early_stopping_patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.early_stopping_patience,
            restore_best_weights=True, verbose=1
        ))

    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss' if val_dataset else 'loss',
        factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ))

    logger.info(f"📊 Multi-view training with {args.fse_channels} FSE channels and {'with' if args.use_decoder_supervision else 'without'} decoder supervision")
    logger.info(f"📊 Training from epoch {initial_epoch} to {initial_epoch + args.epochs}")
    history = model.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=initial_epoch + args.epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_path = os.path.join(LOCAL_TEMP_DIR, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    # Add architecture info to history filename
    arch_info = f"fse{args.fse_channels}"
    if args.use_decoder_supervision:
        arch_info += "_decoder"
    
    hist_blob = bucket.blob(f"{GCS_BASE_PATH}/training_history/history_multiview_{arch_info}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    hist_blob.upload_from_filename(hist_path)

    final_epoch = initial_epoch + args.epochs - 1
    final_step = callbacks[0].step
    save_final_model_to_gcs(model, final_epoch, final_step, history_dict)

    logger.info("✅ Multi-view training completed!")
    return True


# === Entry Point ===
if __name__ == "__main__":
    try:
        if args.multi_view:
            train_multi_view_lithos_model()
        else:
            train_lithos_model()
    except Exception as e:
        logger.error(f"❌ Training crashed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)