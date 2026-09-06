from pathlib import Path


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
