import os
import random
import sys

if "pytest" in sys.argv or "pytest" in sys.modules or os.getenv("CI"):
    print("Setting random seed to 42")
    random.seed(42)
