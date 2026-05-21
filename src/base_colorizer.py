# Mẫu file: src/base_colorizer.py
from abc import ABC, abstractmethod
import numpy as np

class BaseColorizer(ABC):
    """được kế thừa trong colorizer.py ở các module scribble, example, deep"""
    @abstractmethod
    def colorize(self, gray_image: np.ndarray, **kwargs) -> tuple[np.ndarray, dict]:
        """
        - Input: gray_image (H, W, 3) BGR uint8
        - Return: (colorized_bgr, info_dict)
        """
        pass