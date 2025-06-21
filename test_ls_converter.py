#!/usr/bin/env python3
"""
Test label-studio-converter functionality
"""

print("Testing label-studio-converter imports...")

# Test all possible import paths
try:
    from label_studio_converter.utils import rle2mask
    print("✓ Found rle2mask in label_studio_converter.utils")
    rle_func = rle2mask
except ImportError as e:
    print(f"✗ label_studio_converter.utils: {e}")

try:
    from label_studio_converter.brush import rle2mask  
    print("✓ Found rle2mask in label_studio_converter.brush")
    rle_func = rle2mask
except ImportError as e:
    print(f"✗ label_studio_converter.brush: {e}")

# Test what's available in the package
try:
    import label_studio_converter
    print(f"✓ label_studio_converter version info:")
    print(f"  Package location: {label_studio_converter.__file__}")
    print(f"  Available attributes: {dir(label_studio_converter)}")
except ImportError as e:
    print(f"✗ label_studio_converter not available: {e}")

# List what's in the modules
try:
    import label_studio_converter.utils as ls_utils
    print(f"✓ utils module contents: {[x for x in dir(ls_utils) if not x.startswith('_')]}")
except ImportError:
    print("✗ utils module not available")

try:
    import label_studio_converter.brush as ls_brush  
    print(f"✓ brush module contents: {[x for x in dir(ls_brush) if not x.startswith('_')]}")
except ImportError:
    print("✗ brush module not available")

# Also check for other possible modules
try:
    import label_studio_converter.converter as ls_conv
    print(f"✓ converter module contents: {[x for x in dir(ls_conv) if not x.startswith('_')]}")
except ImportError:
    print("✗ converter module not available") 