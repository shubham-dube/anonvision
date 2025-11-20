# blurring/__init__.py
"""
Blurring Module for AnonVision
Provides selective face blurring functionality with multiple blur techniques.
"""

from .blurring_module import (
    FaceBlurrer,
    SelectiveBlurPipeline,
    quick_blur_test
)

__all__ = [
    'FaceBlurrer',
    'SelectiveBlurPipeline',
    'quick_blur_test'
]

__version__ = '1.0.0'