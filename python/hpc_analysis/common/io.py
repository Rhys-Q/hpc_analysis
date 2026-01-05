from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """
    Create the directory if it does not exist and return the path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, content: str) -> Path:
    """
    Write text to a file, ensuring the parent directory exists.
    """
    ensure_dir(path.parent)
    path.write_text(content)
    return path
