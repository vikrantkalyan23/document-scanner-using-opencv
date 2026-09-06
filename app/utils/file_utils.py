from pathlib import Path

from app.config import settings


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"


def get_input_path(filename: str) -> Path:
    """
    Return the full path of an input file.
    """
    return INPUT_DIR / filename


def get_output_path(filename: str) -> Path:
    """
    Return the full output path and ensure
    that the output directory exists.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return OUTPUT_DIR / filename


def list_input_images() -> list[Path]:
    """
    Return supported image files in the input directory.
    """

    if not INPUT_DIR.exists():
        return []

    return sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in settings.SUPPORTED_EXTENSIONS
    )
