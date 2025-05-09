import os
import sys

# Add the scripts directory to the path so that we can import the browse module.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts"))
)
