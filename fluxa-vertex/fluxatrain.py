#!/usr/bin/env python3
# Auralith FLUXA Vertex AI Training Script - Imports and Logging Setup

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
from google.cloud import storage
from tensorflow.keras import regularizers

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# ============================
# Argument Parsing and Globals
# ============================

def parse_args():
    parser = argparse.ArgumentParser(description='Train FLUXA module on Vertex AI')
    
    # Core cloud arguments
    parser.add_argument('--project-id', type=str, default='bright-link-455716-h0')
    parser.add_argument('--bucket-name', type=str, default='auralith')
    parser.add_argument('--base-path', type=str, default='fluxa')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--initial-epoch', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--checkpoint-steps', type=int, default=500)
    parser.add_argument('--repeat-epochs', type=int, default=1)
    
    # Data limits
    parser.add_argument('--max-samples', type=int, default=50000)
    parser.add_argument('--max-val-samples', type=int, default=5000)
    
    # Regularization
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    parser.add_argument('--l2-reg', type=float, default=1e-5)
    
    # Loss weights
    parser.add_argument('--keypoints-weight', type=float, default=0.5)
    parser.add_argument('--segmentation-weight', type=float, default=5.0)
    parser.add_argument('--surface-normals-weight', type=float, default=0.4)
    parser.add_argument('--env-lighting-weight', type=float, default=3.0)
    
    # Controls
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--use-validation', action='store_true')
    parser.add_argument('--skip-checkpoint', action='store_true')
    
    return parser.parse_args()

# Parse args once globally
args = parse_args()

# GCS and local paths
PROJECT_ID = args.project_id
GCS_BUCKET_NAME = args.bucket_name
GCS_BASE_PATH = args.base_path
LOCAL_TEMP_DIR = "/tmp/auralith"

GCS_IMAGES_PATH = f"{GCS_BASE_PATH}/images"
GCS_KEYPOINTS_PATH = f"{GCS_BASE_PATH}/keypoints"
GCS_MASKS_PATH = f"{GCS_BASE_PATH}/segmentation_masks"
GCS_SURFACE_NORMALS_PATH = f"{GCS_BASE_PATH}/surface_normals"
GCS_ENV_LIGHTING_PATH = f"{GCS_BASE_PATH}/environment_lighting"
GCS_CHECKPOINT_DIR = f"{GCS_BASE_PATH}/checkpoints"

GCS_VAL_IMAGES_PATH = f"{GCS_BASE_PATH}/val/images"
GCS_VAL_KEYPOINTS_PATH = f"{GCS_BASE_PATH}/val/keypoints"
GCS_VAL_MASKS_PATH = f"{GCS_BASE_PATH}/val/segmentation_masks"
GCS_VAL_SURFACE_NORMALS_PATH = f"{GCS_BASE_PATH}/val/surface_normals"

# Make local temp dirs
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

# Connect to GCS
try:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    logger.info(f"✅ Connected to GCS bucket '{GCS_BUCKET_NAME}' successfully")
except Exception as e:
    logger.error(f"❌ Failed to connect to GCS: {e}")
    sys.exit(1)
# =====================================
# GCS Sample Discovery + Data Loaders
# =====================================

def list_available_samples(prefix_path, file_extension='.npy'):
    """List samples with matching file type in GCS"""
    try:
        blobs = list(bucket.list_blobs(prefix=f"{prefix_path}/"))
        files = {os.path.splitext(os.path.basename(b.name))[0]
                 for b in blobs if b.name.endswith(file_extension)}
        return files
    except Exception as e:
        logger.error(f"❌ Error listing from {prefix_path}: {e}")
        return set()

def find_common_samples(training=True):
    """Find samples that have image, keypoints, mask, normals"""
    if training:
        kp = list_available_samples(GCS_KEYPOINTS_PATH, '.npy')
        mask = list_available_samples(GCS_MASKS_PATH, '.png')
        normals = list_available_samples(GCS_SURFACE_NORMALS_PATH, '.npy')
    else:
        kp = list_available_samples(GCS_VAL_KEYPOINTS_PATH, '.npy')
        mask = list_available_samples(GCS_VAL_MASKS_PATH, '.png')
        normals = list_available_samples(GCS_VAL_SURFACE_NORMALS_PATH, '.npy')

    return list(kp & mask & normals)
def download_image_from_gcs(image_id, training=True):
    prefix = GCS_IMAGES_PATH if training else GCS_VAL_IMAGES_PATH
    for ext in ['.jpg', '.png']:
        blob_path = f"{prefix}/{image_id}{ext}"
        blob = bucket.blob(blob_path)
        if blob.exists():
            try:
                image_bytes = blob.download_as_bytes()
                img = tf.image.decode_image(image_bytes, channels=3)
                img = tf.image.resize(img, (480, 640))
                return tf.cast(img, tf.float32) / 255.0
            except Exception as e:
                logger.warning(f"⚠️ Error decoding image {image_id}: {e}")
    return tf.zeros((480, 640, 3), dtype=tf.float32)

def load_keypoints_from_gcs(image_id, training=True):
    prefix = GCS_KEYPOINTS_PATH if training else GCS_VAL_KEYPOINTS_PATH
    blob = bucket.blob(f"{prefix}/{image_id}.npy")
    if blob.exists():
        keypoints_bytes = blob.download_as_bytes()
        keypoints = np.load(io.BytesIO(keypoints_bytes))
        return tf.convert_to_tensor(keypoints, dtype=tf.float32)
    return tf.zeros((480, 640, 17), dtype=tf.float32)

def load_segmentation_mask_from_gcs(image_id, training=True):
    prefix = GCS_MASKS_PATH if training else GCS_VAL_MASKS_PATH
    blob = bucket.blob(f"{prefix}/{image_id}.png")
    if blob.exists():
        try:
            mask_bytes = blob.download_as_bytes()
            mask = tf.image.decode_png(mask_bytes, channels=1)
            mask = tf.image.resize(mask, (480, 640))
            return tf.cast(mask, tf.float32) / 255.0
        except Exception as e:
            logger.warning(f"⚠️ Mask decoding error for {image_id}: {e}")
    return tf.zeros((480, 640, 1), dtype=tf.float32)

def load_surface_normals_from_gcs(image_id, training=True):
    prefix = GCS_SURFACE_NORMALS_PATH if training else GCS_VAL_SURFACE_NORMALS_PATH
    blob = bucket.blob(f"{prefix}/{image_id}.npy")
    if blob.exists():
        normals_bytes = blob.download_as_bytes()
        normals = np.load(io.BytesIO(normals_bytes))
        if normals.shape != (480, 640, 3):
            normals = cv2.resize(normals, (640, 480))
        min_val, max_val = np.min(normals), np.max(normals)
        if max_val > 1.0 or min_val < -1.0:
            normals = 2.0 * (normals - min_val) / (max_val - min_val + 1e-8) - 1.0
        return tf.convert_to_tensor(normals, dtype=tf.float32)
    return tf.ones((480, 640, 3), dtype=tf.float32) * tf.constant([0.0, 0.0, 1.0])
def load_environment_lighting_from_gcs():
    """Load random environment lighting (9 SH coefficients)"""
    try:
        blobs = list(bucket.list_blobs(prefix=GCS_ENV_LIGHTING_PATH, max_results=100))
        env_files = [b for b in blobs if b.name.endswith('.json')]
        if not env_files:
            return tf.random.normal((9,), mean=0.5, stddev=0.1)
        random_env = random.choice(env_files)
        env_json = json.loads(random_env.download_as_string())
        sh = np.mean(np.array(env_json['spherical_harmonics']), axis=0)
        sh = (sh - sh.min()) / (sh.max() - sh.min() + 1e-8)
        return tf.convert_to_tensor(sh, dtype=tf.float32)
    except Exception as e:
        logger.warning(f"⚠️ Env lighting load error: {e}")
        return tf.random.normal((9,), mean=0.5, stddev=0.1)
def apply_augmentation(image, keypoints, segmentation, surface_normals):
    if tf.random.uniform(()) > 0.5:
        return image, keypoints, segmentation, surface_normals

    flip_lr = tf.random.uniform(()) > 0.5
    brightness = tf.random.uniform((), -0.1, 0.1)
    contrast = tf.random.uniform((), 0.9, 1.1)

    if flip_lr:
        image = tf.image.flip_left_right(image)
        keypoints = tf.image.flip_left_right(keypoints)
        segmentation = tf.image.flip_left_right(segmentation)
        surface_normals = tf.image.flip_left_right(surface_normals)
        surface_normals = tf.stack([
            -surface_normals[..., 0],  # flip X
            surface_normals[..., 1],
            surface_normals[..., 2]
        ], axis=-1)

    image = tf.image.adjust_brightness(image, brightness)
    image = tf.image.adjust_contrast(image, contrast)
    return tf.clip_by_value(image, 0.0, 1.0), keypoints, segmentation, surface_normals
def data_generator(sample_ids, batch_size=8, training=True):
    samples = sample_ids.copy()
    def _epoch():
        if training:
            random.shuffle(samples)
        for i in range(0, len(samples), batch_size):
            batch_ids = samples[i:i+batch_size]
            imgs, kps, masks, norms, lights = [], [], [], [], []

            for img_id in batch_ids:
                try:
                    img = download_image_from_gcs(img_id, training)
                    kp = load_keypoints_from_gcs(img_id, training)
                    mask = load_segmentation_mask_from_gcs(img_id, training)
                    norm = load_surface_normals_from_gcs(img_id, training)

                    if training:
                        img, kp, mask, norm = apply_augmentation(img, kp, mask, norm)

                    imgs.append(img)
                    kps.append(kp)
                    masks.append(mask)
                    norms.append(norm)
                except Exception as e:
                    logger.warning(f"⚠️ Failed loading {img_id}: {e}")

            if not imgs:
                continue

            env_light = load_environment_lighting_from_gcs()
            lights = [env_light] * len(imgs)

            yield tf.stack(imgs), {
                'fluxa_keypoints': tf.stack(kps),
                'fluxa_segmentation': tf.stack(masks),
                'fluxa_surface_normals': tf.stack(norms),
                'fluxa_environment_lighting': tf.stack(lights)
            }

    while True:
        yield from _epoch()
def create_fluxa_dataset(sample_ids, batch_size=8, training=True):
    return tf.data.Dataset.from_generator(
        lambda: data_generator(sample_ids, batch_size=batch_size, training=training),
        output_signature=(
            tf.TensorSpec(shape=(None, 480, 640, 3), dtype=tf.float32),
            {
                'fluxa_keypoints': tf.TensorSpec(shape=(None, 480, 640, 17), dtype=tf.float32),
                'fluxa_segmentation': tf.TensorSpec(shape=(None, 480, 640, 1), dtype=tf.float32),
                'fluxa_surface_normals': tf.TensorSpec(shape=(None, 480, 640, 3), dtype=tf.float32),
                'fluxa_environment_lighting': tf.TensorSpec(shape=(None, 9), dtype=tf.float32)
            }
        )
    ).prefetch(tf.data.AUTOTUNE)
class KeypointAccuracy(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.05, name='keypoint_accuracy', **kwargs):
        super(KeypointAccuracy, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total_keypoints = self.add_weight(name='total_keypoints', initializer='zeros')
        self.correct_keypoints = self.add_weight(name='correct_keypoints', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        abs_diff = tf.abs(y_true - y_pred)
        distances = tf.reduce_mean(abs_diff, axis=[1, 2])  # (batch, 17)
        correct = tf.cast(distances < self.threshold, tf.float32)
        self.total_keypoints.assign_add(tf.cast(tf.size(correct), tf.float32))
        self.correct_keypoints.assign_add(tf.reduce_sum(correct))

    def result(self):
        return self.correct_keypoints / (self.total_keypoints + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total_keypoints.assign(0)
        self.correct_keypoints.assign(0)
class AuralithNeuralNet:
    def __init__(self,
                 input_shape=(480, 640, 3),
                 modules_active={"fluxa": True, "prism": False, "lithos": False},
                 backbone_type="mobilenet_v2",
                 training=False,
                 learning_rate=0.001,
                 dropout_rate=0.2,
                 l2_reg=1e-5,
                 loss_weights=None,
                 weights_path=None):
        
        self.input_shape = input_shape
        self.modules_active = modules_active
        self.backbone_type = backbone_type
        self.training = training
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        self.loss_weights = loss_weights or {
            "fluxa_keypoints": 0.5,
            "fluxa_segmentation": 5.0,
            "fluxa_surface_normals": 0.4,
            "fluxa_environment_lighting": 3.0
        }

        self._build_model()

        if weights_path:
            self._try_load_weights(weights_path)

        if self.training:
            self._compile_model()

    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape, name="image_input")
        backbone = self._build_backbone(inputs)
        outputs = {}

        if self.modules_active["fluxa"]:
            outputs.update(self._build_fluxa_branch(backbone))
        if self.modules_active["prism"]:
            outputs.update(self._build_prism_branch(backbone))
        if self.modules_active["lithos"]:
            outputs.update(self._build_lithos_branch(backbone))

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="auralith_core")
    def _build_backbone(self, inputs):
        if self.backbone_type == "mobilenet_v2":
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet",
                alpha=0.75
            )
            base_model.trainable = self.training
            return base_model(inputs)

        elif self.backbone_type == "efficientnet_lite":
            base_model = tf.keras.applications.EfficientNetB0(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet"
            )
            base_model.trainable = self.training
            return base_model(inputs)

        else:
            # Custom fallback
            x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same',
                                       kernel_regularizer=regularizers.l2(self.l2_reg))(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            x = self._inverted_res_block(x, filters=64, strides=2, expansion=6)
            x = self._inverted_res_block(x, filters=96, strides=2, expansion=6)
            x = self._inverted_res_block(x, filters=160, strides=2, expansion=6)

            return x
    def _inverted_res_block(self, x, filters, strides, expansion):
        shortcut = x

        x = tf.keras.layers.Conv2D(expansion * x.shape[-1], kernel_size=1,
                                   kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same',
                                            kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters, kernel_size=1,
                                   kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if strides == 1 and x.shape[-1] == shortcut.shape[-1]:
            x = tf.keras.layers.Add()([x, shortcut])

        return x
    def _build_fluxa_branch(self, backbone):
        x = tf.keras.layers.Conv2D(128, kernel_size=1,
                                   kernel_regularizer=regularizers.l2(self.l2_reg))(backbone)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        for i in range(4):
            x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
            x = tf.keras.layers.Conv2D(96 if i < 2 else 64, kernel_size=3, padding='same',
                                       kernel_regularizer=regularizers.l2(self.l2_reg))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            if i == 2:
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        # Final upsample to input resolution
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same',
                                   kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # Output 1: Keypoints
        keypoints = tf.keras.layers.Conv2D(17, kernel_size=1, name='fluxa_keypoints')(x)

        # Output 2: Segmentation
        seg = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same',
                                     kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        seg = tf.keras.layers.BatchNormalization()(seg)
        seg = tf.keras.layers.ReLU()(seg)
        seg = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid', name='fluxa_segmentation')(seg)

        # Output 3: Surface normals
        norm = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same',
                                      kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        norm = tf.keras.layers.BatchNormalization()(norm)
        norm = tf.keras.layers.ReLU()(norm)
        norm = tf.keras.layers.Conv2D(3, kernel_size=1, activation='tanh', name='fluxa_surface_normals')(norm)

        # Output 4: Environment lighting
        env = tf.keras.layers.GlobalAveragePooling2D()(x)
        env = tf.keras.layers.Dense(32, activation='relu',
                                    kernel_regularizer=regularizers.l2(self.l2_reg))(env)
        env = tf.keras.layers.Dropout(self.dropout_rate)(env)
        env = tf.keras.layers.Dense(9, name='fluxa_environment_lighting')(env)

        return {
            'fluxa_keypoints': keypoints,
            'fluxa_segmentation': seg,
            'fluxa_surface_normals': norm,
            'fluxa_environment_lighting': env
        }
    def _build_prism_branch(self, backbone):
        x = tf.keras.layers.Conv2D(128, kernel_size=1,
                                   kernel_regularizer=regularizers.l2(self.l2_reg))(backbone)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(96, kernel_size=3, padding='same',
                                   kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        lighting = tf.keras.layers.GlobalAveragePooling2D()(x)
        lighting = tf.keras.layers.Dense(64, activation='relu')(lighting)
        lighting = tf.keras.layers.Dropout(self.dropout_rate)(lighting)
        lighting = tf.keras.layers.Dense(16, name='prism_lighting_application')(lighting)

        deform = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
        deform = tf.keras.layers.BatchNormalization()(deform)
        deform = tf.keras.layers.ReLU()(deform)
        deform = tf.keras.layers.Conv2D(2, kernel_size=1, name='prism_deformation')(deform)

        shadow = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same')(x)
        shadow = tf.keras.layers.BatchNormalization()(shadow)
        shadow = tf.keras.layers.ReLU()(shadow)
        shadow = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid', name='prism_shadow')(shadow)

        return {
            'prism_lighting_application': lighting,
            'prism_deformation': deform,
            'prism_shadow': shadow
        }

    def _build_lithos_branch(self, backbone):
        x = tf.keras.layers.Conv2D(128, kernel_size=1,
                                   kernel_regularizer=regularizers.l2(self.l2_reg))(backbone)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        base = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
        base = tf.keras.layers.BatchNormalization()(base)
        base = tf.keras.layers.ReLU()(base)
        base = tf.keras.layers.Conv2D(3, kernel_size=1, activation='sigmoid', name='lithos_base_encoding')(base)

        return {
            'lithos_base_encoding': base
        }
    def _inverted_res_block(self, x, filters, strides, expansion):
        shortcut = x
        x = tf.keras.layers.Conv2D(expansion * x.shape[-1], kernel_size=1,
                                   kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same',
                                            kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters, kernel_size=1,
                                   kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if strides == 1 and shortcut.shape[-1] == filters:
            x = tf.keras.layers.Add()([shortcut, x])
        return x

    def _compile_model(self):
        losses = {
            "fluxa_keypoints": "mse",
            "fluxa_segmentation": "binary_crossentropy",
            "fluxa_surface_normals": "mse",
            "fluxa_environment_lighting": "mse"
        }

        loss_weights = {
            "fluxa_keypoints": self.loss_weights.get("fluxa_keypoints", 0.5),
            "fluxa_segmentation": self.loss_weights.get("fluxa_segmentation", 5.0),
            "fluxa_surface_normals": self.loss_weights.get("fluxa_surface_normals", 0.4),
            "fluxa_environment_lighting": self.loss_weights.get("fluxa_environment_lighting", 3.0)
        }

        metrics = {
            "fluxa_keypoints": ["mae"],
            "fluxa_segmentation": ["accuracy"],
            "fluxa_surface_normals": ["mae"],
            "fluxa_environment_lighting": ["mae"]
        }

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
    def _try_load_weights(self, weights_path):
        try:
            logger.info("📦 Building model with dummy input before loading weights...")
            dummy_input = tf.zeros((1, *self.input_shape))
            _ = self.model(dummy_input)

            logger.info(f"📥 Loading weights from: {weights_path}")
            try:
                self.model.load_weights(weights_path)
                logger.info(f"✅ Weights loaded successfully (strict mode)")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Strict weight loading failed: {e}")

                logger.info("🔁 Attempting flexible loading with skip_mismatch=True...")
                load_status = self.model.load_weights(
                    weights_path, skip_mismatch=True, by_name=True
                )
                load_status.expect_partial()
                logger.info(f"✅ Weights loaded partially with skip_mismatch")
                return True

        except Exception as e:
            logger.error(f"❌ Error during weight loading: {e}")
            return False
    def inference(self, image):
        """Run inference on a single image"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return self.model.predict(image)

    def save_weights(self, path):
        """Save model weights"""
        self.model.save_weights(path)

    def load_weights(self, path):
        """Load model weights"""
        self._try_load_weights(path)

    def export_model(self, path, format="saved_model"):
        """Export the model for deployment"""
        if format == "saved_model":
            self.model.save(path)
        elif format == "tflite":
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            with open(path, 'wb') as f:
                f.write(tflite_model)
        else:
            raise ValueError(f"Unsupported export format: {format}")
class GCSCheckpointCallback(tf.keras.callbacks.Callback):
    """Custom callback to save checkpoints to GCS"""

    def __init__(self, checkpoint_dir, gcs_dir, save_freq='epoch'):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.gcs_dir = gcs_dir
        self.save_freq = save_freq
        self.step_counter = 0
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        if self.save_freq == 'epoch':
            self._save_checkpoint(epoch, logs)

    def on_batch_end(self, batch, logs=None):
        if isinstance(self.save_freq, int):
            self.step_counter += 1
            if self.step_counter % self.save_freq == 0:
                epoch = self.params.get('epoch', 0)
                self._save_checkpoint(epoch, logs, step=self.step_counter)

    def _save_checkpoint(self, epoch, logs=None, step=None):
        try:
            filename = (
                f"checkpoint-{epoch+1:03d}-step{step:05d}.weights.h5"
                if step is not None else
                f"checkpoint-{epoch+1:03d}.weights.h5"
            )
            local_path = os.path.join(self.checkpoint_dir, filename)
            gcs_path = os.path.join(self.gcs_dir, filename)

            self.model.save_weights(local_path)
            bucket.blob(gcs_path).upload_from_filename(local_path)
            logger.info(f"✅ Checkpoint saved to GCS: {gcs_path}")

            # Save "latest_model"
            latest_path = os.path.join(self.checkpoint_dir, "latest_model.weights.h5")
            latest_gcs = os.path.join(self.gcs_dir, "latest_model.weights.h5")
            tf.io.gfile.copy(local_path, latest_path, overwrite=True)
            bucket.blob(latest_gcs).upload_from_filename(latest_path)
            logger.info(f"✅ Latest model saved to GCS: {latest_gcs}")

            # Save best model by val_loss
            if logs and 'val_loss' in logs:
                val_loss = logs['val_loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_local = os.path.join(self.checkpoint_dir, "best_model.weights.h5")
                    best_gcs = os.path.join(self.gcs_dir, "best_model.weights.h5")
                    tf.io.gfile.copy(local_path, best_local, overwrite=True)
                    bucket.blob(best_gcs).upload_from_filename(best_local)
                    logger.info(f"🏅 Best model updated on GCS: {best_gcs} (val_loss: {val_loss:.4f})")

        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    """Custom TensorBoard callback that uploads logs to GCS"""

    def __init__(self, log_dir, gcs_log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.gcs_log_dir = gcs_log_dir

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        try:
            for root, _, files in os.walk(self.log_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, self.log_dir)
                    gcs_path = os.path.join(self.gcs_log_dir, rel_path)
                    bucket.blob(gcs_path).upload_from_filename(local_path)
            logger.info(f"📤 TensorBoard logs uploaded to GCS: {self.gcs_log_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to upload TensorBoard logs: {e}")
class MetricTracker(tf.keras.callbacks.Callback):
    """Custom callback to track individual metrics and detect overfitting"""

    def __init__(self, metrics_to_track=None, patience=5):
        super().__init__()
        self.metrics_to_track = metrics_to_track or [
            'fluxa_keypoints_mae', 
            'fluxa_segmentation_accuracy', 
            'fluxa_surface_normals_mae'
        ]
        self.patience = patience
        self.history = {metric: [] for metric in self.metrics_to_track}
        self.val_history = {f'val_{metric}': [] for metric in self.metrics_to_track}
        self.no_improvement = {metric: 0 for metric in self.metrics_to_track}
        self.best_values = {metric: float('inf') for metric in self.metrics_to_track}
        self.best_values['fluxa_segmentation_accuracy'] = 0  # Higher is better for accuracy

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for metric in self.metrics_to_track:
            if metric in logs:
                self.history[metric].append(logs[metric])

            val_metric = f'val_{metric}'
            if val_metric in logs:
                self.val_history[val_metric].append(logs[val_metric])
                current = logs[val_metric]

                # Accuracy (higher is better)
                if metric == 'fluxa_segmentation_accuracy':
                    if current > self.best_values[metric]:
                        self.best_values[metric] = current
                        self.no_improvement[metric] = 0
                    else:
                        self.no_improvement[metric] += 1
                else:
                    if current < self.best_values[metric]:
                        self.best_values[metric] = current
                        self.no_improvement[metric] = 0
                    else:
                        self.no_improvement[metric] += 1

                if self.no_improvement[metric] == 0:
                    logger.info(f"✅ New best {val_metric}: {current:.4f}")
                else:
                    logger.info(f"⚠️ No improvement in {val_metric} for {self.no_improvement[metric]} epochs")

                if self.no_improvement[metric] >= self.patience:
                    logger.warning(f"❌ Potential overfitting detected for {metric}!")
def train_fluxa_module(sample_ids, val_ids=None, batch_size=16, val_batch_size=32, epochs=50, initial_epoch=0, checkpoint_steps=500):
    """Train the FLUXA module with checkpoints to GCS and validation"""
    logger.info("🔧 Setting up FLUXA training...")

    # Validation logic
    if val_ids is None and args.use_validation:
        val_ids = find_common_samples(training=False)
        if not val_ids:
            logger.info("📁 No separate validation data found. Splitting training set...")
            train_size = int(0.9 * len(sample_ids))
            val_ids = sample_ids[train_size:]
            sample_ids = sample_ids[:train_size]
    elif val_ids is None:
        train_size = int(0.9 * len(sample_ids))
        val_ids = sample_ids[train_size:]
        sample_ids = sample_ids[:train_size]

    if val_ids and len(val_ids) > args.max_val_samples:
        logger.info(f"⚠️ Limiting validation to {args.max_val_samples} samples")
        val_ids = val_ids[:args.max_val_samples]

    logger.info(f"🔢 Training with {len(sample_ids)} samples, validating with {len(val_ids)} samples")

    # Datasets
    train_dataset = create_fluxa_dataset(sample_ids, batch_size, training=True)
    val_dataset = create_fluxa_dataset(val_ids, val_batch_size, training=False)

    # Initialize model
    fluxa_model = AuralithNeuralNet(
        input_shape=(480, 640, 3),
        modules_active={"fluxa": True, "prism": False, "lithos": False},
        training=True,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg,
        loss_weights={
            "fluxa_keypoints": args.keypoints_weight,
            "fluxa_segmentation": args.segmentation_weight,
            "fluxa_surface_normals": args.surface_normals_weight,
            "fluxa_environment_lighting": args.env_lighting_weight
        }
    )

    # Load checkpoint if not skipped
    initial_epoch = args.initial_epoch
    if not args.skip_checkpoint:
        logger.info("🔍 Looking for the most recent checkpoint in GCS...")
        checkpoint_blobs = list(bucket.list_blobs(prefix=f"{GCS_CHECKPOINT_DIR}/"))
        checkpoint_files = [b for b in checkpoint_blobs if b.name.endswith('.weights.h5')]

    if checkpoint_files and args.initial_epoch == 0:  # Only auto-load if not manually set
        checkpoint_files.sort(key=lambda x: x.updated, reverse=True)
        most_recent = checkpoint_files[0]

        # Try to extract epoch from filename
        try:
            filename = os.path.basename(most_recent.name)
            if 'checkpoint-' in filename:
                epoch_str = filename.split('checkpoint-')[1].split('-')[0]
                if epoch_str.isdigit():
                    initial_epoch = int(epoch_str)
        except (IndexError, ValueError):
            logger.warning("⚠️ Could not parse epoch number from checkpoint name.")

        logger.info(f"📥 Loading most recent checkpoint: {most_recent.name}")
        local_checkpoint_path = f"{LOCAL_TEMP_DIR}/latest_checkpoint.weights.h5"
        most_recent.download_to_filename(local_checkpoint_path)

        try:
            _ = fluxa_model.model(tf.zeros((1, 480, 640, 3)))  # Build model
            fluxa_model.load_weights(local_checkpoint_path)
            logger.info(f"✅ Checkpoint '{most_recent.name}' loaded successfully (starting from epoch {initial_epoch})")
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            initial_epoch = args.initial_epoch
    else:
        if args.initial_epoch > 0:
            logger.info(f"🔄 Starting from specified epoch: {args.initial_epoch}")
            initial_epoch = args.initial_epoch
        else:
            logger.info("⚠️ No checkpoints found. Training from scratch.")
            initial_epoch = 0

    # Directories
    log_dir = os.path.join(LOCAL_TEMP_DIR, "logs", datetime.now().strftime('%Y%m%d-%H%M%S'))
    checkpoint_dir = os.path.join(LOCAL_TEMP_DIR, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    gcs_log_dir = os.path.join(GCS_BASE_PATH, "logs", datetime.now().strftime('%Y%m%d-%H%M%S'))

    # Callbacks
    callbacks = [
        GCSCheckpointCallback(checkpoint_dir, GCS_CHECKPOINT_DIR, save_freq=checkpoint_steps),
        CustomTensorBoard(log_dir=log_dir, gcs_log_dir=gcs_log_dir),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience, restore_best_weights=True),
        MetricTracker(patience=args.early_stopping_patience // 2)
    ]

    # Steps
    steps_per_epoch = max(1, len(sample_ids) // batch_size // args.repeat_epochs)
    validation_steps = max(1, len(val_ids) // val_batch_size)

    logger.info(f"🧠 Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

    # Train
    history = fluxa_model.model.fit(
        train_dataset,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        epochs=epochs * args.repeat_epochs,
        initial_epoch=initial_epoch * args.repeat_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )

    # Final save
    final_path = os.path.join(LOCAL_TEMP_DIR, "final_model.weights.h5")
    fluxa_model.save_weights(final_path)
    bucket.blob(GCS_CHECKPOINT_DIR + "/final_model.weights.h5").upload_from_filename(final_path)
    logger.info("💾 Final weights saved to GCS")

    try:
        tflite_path = os.path.join(LOCAL_TEMP_DIR, "fluxa_module.tflite")
        fluxa_model.export_model(tflite_path, format="tflite")
        bucket.blob(GCS_BASE_PATH + "/models/fluxa_module.tflite").upload_from_filename(tflite_path)
        logger.info("📦 TFLite model exported to GCS")
    except Exception as e:
        logger.error(f"❌ Failed to export TFLite: {e}")

    return history
# === Main Entry Point ===
def main():
    """Main training function for Vertex AI"""
    try:
        logger.info("🚀 Launching FLUXA Vertex AI training pipeline")
        
        # List all complete training samples
        training_samples = find_common_samples(training=True)
        validation_samples = find_common_samples(training=False) if args.use_validation else None

        # Limit training sample count
        if len(training_samples) > args.max_samples:
            logger.info(f"⚠️ Trimming training samples from {len(training_samples)} → {args.max_samples}")
            training_samples = training_samples[:args.max_samples]

        # Shuffle
        random.shuffle(training_samples)

        # Train
        history = train_fluxa_module(
            sample_ids=training_samples,
            val_ids=validation_samples,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            epochs=args.epochs,
            initial_epoch=args.initial_epoch,
            checkpoint_steps=args.checkpoint_steps
        )

        logger.info("✅ Training complete!")

        return 0
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
