# adv_skin_cancer/src/__init__.py
"""
Skin Cancer Classification Package
Advance Research Methodologies Winter Semester 2025: Class-Aware Multifaceted Approach for 
Imbalanced Dermoscopic Image Classification

Author: Mati Nakphon
University: University of Europe for Applied Sciences
"""

__version__ = "1.0.0"
__author__ = "Mati Nakphon"
__email__ = "mati.nakphon@ue-germany.de"
__license__ = "Academic Use Only"

# Core exports
from .config import Config, config
__all__ = ['Config', 'config']

print(f"✓ SkinCancerClassification v{__version__}")
print("Note: Other modules will be loaded as they are created")