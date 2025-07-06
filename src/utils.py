import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle.

    Args:
        file_path (str or Path): The path to save the file.
        obj (Any): The Python object to serialize.
    """
    try:
        dir_path = Path(file_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a pickled object from disk.

    Args:
        file_path (str or Path): The path to the pickled file.

    Returns:
        Any: The deserialized Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
