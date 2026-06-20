"""Launch the artifact-only Streamlit recommendation demo."""

import subprocess
import sys


def main() -> None:
    command = [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
