"""
Pytest configuration: put the repo root on sys.path so every test can import
the tvopt_analysis package without requiring a full pip install.
"""

import sys
import os

# Make the repo root importable for every test in this suite.
_root = os.path.dirname(os.path.dirname(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
