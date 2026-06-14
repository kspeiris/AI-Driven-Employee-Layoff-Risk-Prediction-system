import sys
from pathlib import Path

# Ensure the project root (this folder) is on sys.path so we can import the dashboard
sys.path.append(str(Path(__file__).parent))

# Importing the module runs the Streamlit app defined in `dashboard/Home.py`
import dashboard.Home
