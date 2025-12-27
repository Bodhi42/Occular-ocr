"""Utilities for loading model weights from local files"""

import os
import torch
from pathlib import Path
from typing import Optional


# Path to local weights directory
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

# Local weights filenames
CRAFT_WEIGHTS_FILE = "craft_mlt_25k.pth"
CRNN_WEIGHTS_FILE = "crnn.pth"
VITSTR_WEIGHTS_FILE = "vitstr_small.pth"
DBNET_WEIGHTS_FILE = "dbnet_resnet18.pth"


def get_weights_path(filename: str) -> Path:
    """
    Get path to local weights file

    Args:
        filename: weights filename

    Returns:
        Path to weights file
    """
    return WEIGHTS_DIR / filename


def load_craft_weights(model, device='cpu', weights_path: Optional[Path] = None):
    """
    Load pretrained CRAFT weights from local file

    Args:
        model: CRAFT model instance
        device: device to load weights to
        weights_path: custom path to weights file (optional)

    Returns:
        model with loaded weights
    """
    if weights_path is None:
        weights_path = get_weights_path(CRAFT_WEIGHTS_FILE)

    if not weights_path.exists():
        print(f"Warning: CRAFT weights not found at {weights_path}")
        print("Using randomly initialized weights")
        return model

    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)

        # Remove 'module.' prefix if present (from DataParallel)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded CRAFT weights from {weights_path}")
    except Exception as e:
        print(f"Failed to load CRAFT weights: {e}")
        print("Using randomly initialized weights")

    return model


def remap_crnn_keys(state_dict):
    """
    Remap CRNN checkpoint keys to match current model architecture

    Checkpoint uses: cnn.conv0, cnn.batchnorm2, rnn.0.embedding
    Model uses: cnn.0, cnn.7, rnn.0.linear

    Args:
        state_dict: original state dict from checkpoint

    Returns:
        remapped state dict
    """
    # CNN layer mapping: named layers -> Sequential indices
    cnn_mapping = {
        'conv0': '0',   # Conv2d(1, 64)
        'conv1': '3',   # Conv2d(64, 128)
        'conv2': '6',   # Conv2d(128, 256)
        'batchnorm2': '7',  # BatchNorm2d(256)
        'conv3': '9',   # Conv2d(256, 256)
        'conv4': '12',  # Conv2d(256, 512)
        'batchnorm4': '13',  # BatchNorm2d(512)
        'conv5': '15',  # Conv2d(512, 512)
        'conv6': '18',  # Conv2d(512, 512)
        'batchnorm6': '19',  # BatchNorm2d(512)
    }

    remapped = {}
    for key, value in state_dict.items():
        new_key = key

        # Remap CNN layers
        if key.startswith('cnn.'):
            parts = key.split('.')
            if len(parts) >= 2:
                old_layer_name = parts[1]
                if old_layer_name in cnn_mapping:
                    parts[1] = cnn_mapping[old_layer_name]
                    new_key = '.'.join(parts)

        # Remap RNN embedding -> linear
        if '.embedding.' in key:
            new_key = key.replace('.embedding.', '.linear.')

        remapped[new_key] = value

    return remapped


def load_crnn_weights(model, device='cpu', weights_path: Optional[Path] = None):
    """
    Load pretrained CRNN weights from local file

    Args:
        model: CRNN model instance
        device: device to load weights to
        weights_path: custom path to weights file (optional)

    Returns:
        model with loaded weights
    """
    if weights_path is None:
        weights_path = get_weights_path(CRNN_WEIGHTS_FILE)

    if not weights_path.exists():
        print(f"Warning: CRNN weights not found at {weights_path}")
        print("Using randomly initialized weights")
        return model

    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)

        # Remove 'module.' prefix if present (from DataParallel)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Remap keys to match current architecture
        state_dict = remap_crnn_keys(state_dict)

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded CRNN weights from {weights_path}")
    except Exception as e:
        print(f"Failed to load CRNN weights: {e}")
        print("Using randomly initialized weights")

    return model


def load_vitstr_weights(model, device='cpu', weights_path: Optional[Path] = None):
    """
    Load pretrained ViTSTR weights from local file

    Args:
        model: ViTSTR model instance
        device: device to load weights to
        weights_path: custom path to weights file (optional)

    Returns:
        model with loaded weights
    """
    if weights_path is None:
        weights_path = get_weights_path(VITSTR_WEIGHTS_FILE)

    if not weights_path.exists():
        print(f"Warning: ViTSTR weights not found at {weights_path}")
        print("Using randomly initialized weights")
        return model

    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)

        # Remove 'module.' prefix if present (from DataParallel)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded ViTSTR weights from {weights_path}")
    except Exception as e:
        print(f"Failed to load ViTSTR weights: {e}")
        print("Using randomly initialized weights")

    return model


def load_dbnet_weights(model, device='cpu', weights_path: Optional[Path] = None):
    """
    Load pretrained DBNet weights from local file

    Args:
        model: DBNet model instance
        device: device to load weights to
        weights_path: custom path to weights file (optional)

    Returns:
        model with loaded weights
    """
    if weights_path is None:
        weights_path = get_weights_path(DBNET_WEIGHTS_FILE)

    if not weights_path.exists():
        print(f"Warning: DBNet weights not found at {weights_path}")
        print("Using ImageNet pretrained backbone only")
        return model

    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)

        # Remove 'module.' prefix if present (from DataParallel)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded DBNet weights from {weights_path}")
    except Exception as e:
        print(f"Failed to load DBNet weights: {e}")
        print("Using ImageNet pretrained backbone only")

    return model
