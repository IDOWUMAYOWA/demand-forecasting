import yaml
from pathlib import Path
from typing import Union


def read_config(path: Union[str, Path]) -> dict:
    """
    Reads a YAML configuration file and returns it as a dictionary.

    Parameters:
    ----------
    path : str or Path
        Path to the YAML file.

    Returns:
    -------
    dict
        Parsed YAML content as a dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f)