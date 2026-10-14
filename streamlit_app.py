"""
Root entrypoint for Streamlit Community Cloud.
Forwards execution to app/streamlit_app.py.
"""

import sys
from pathlib import Path
import runpy

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

target_app = PROJECT_ROOT / "app" / "streamlit_app.py"
runpy.run_path(str(target_app), run_name="__main__")
