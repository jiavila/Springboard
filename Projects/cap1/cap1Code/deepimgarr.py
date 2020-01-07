"""
DeepImgArray
------------

A class for applying manipulations to deep learning arrays.
"""

import numpy as np


class DeepImgArray():

    def __init__(self, num_images: int = None,
                 num_rows: int = None,
                 num_cols: int = None,
                 num_channels: int = None):
        if num_images and num_rows and num_cols and num_channels:
            self.DataArr = np.empty(shape=(num_images, num_rows, num_cols,
                                           num_channels))

