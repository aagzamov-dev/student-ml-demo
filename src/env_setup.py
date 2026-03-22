import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    if "LOKY_MAX_CPU_COUNT" not in os.environ:
        cpu_count = os.cpu_count() or 1
        os.environ["LOKY_MAX_CPU_COUNT"] = str(cpu_count)
