import sys
import numpy as np
import pytest
sys.path.insert(0, "build")

import fast_blur
from python_impl.blur import box_blur

def test_output_shape():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = fast_blur.blur(img, radius=3)
    assert result.shape == (64, 64, 3)

def test_output_dtype():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = fast_blur.blur(img, radius=3)
    assert result.dtype == np.uint8

def test_matches_python_baseline():
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    flat = img.flatten().tobytes()

    py_out = np.frombuffer(box_blur(flat, 32, 32, 3, 3), dtype=np.uint8).reshape(32, 32, 3)
    cpp_out = fast_blur.blur(img, radius=3)

    # allow ±1 difference due to integer rounding
    assert np.max(np.abs(cpp_out.astype(int) - py_out.astype(int))) <= 1

def test_single_pixel_image():
    img = np.array([[[128, 64, 32]]], dtype=np.uint8)
    result = fast_blur.blur(img, radius=5)
    assert result.shape == (1, 1, 3)

def test_grayscale_image():
    img = np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)
    result = fast_blur.blur(img, radius=3)
    assert result.shape == (64, 64, 1)
