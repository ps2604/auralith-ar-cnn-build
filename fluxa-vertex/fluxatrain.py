# file: fluxatrain.py (Cleaned Version)
# Removed all GCP/GCS dependencies for local research archive.

import os
import sys
import logging
import argparse
import tensorflow as tf

logger = logging.getLogger("FLUXA_CNN_TRAINING")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    
    # Local-only initialization
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger.info(f"✅ Local training initialized: {args.checkpoint_dir}")

if __name__ == "__main__":
    main()
