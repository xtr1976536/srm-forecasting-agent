"""Canonical Streamlit Cloud entrypoint.

The public deployment must select this root-level file. It forwards to the
package application so Streamlit's execution context is unambiguous.
"""
import sys
from pathlib import Path

_repo_parent = str(Path(__file__).resolve().parent.parent)
if _repo_parent not in sys.path:
    sys.path.insert(0, _repo_parent)

from srm_agent.streamlit_app import *  # noqa: F401,F403,E402
