"""
DeepImgArray
------------

A class for applying manipulations to deep learning arrays.
"""

import numpy as np
import abc
from skimage import exposure as xp

class AbstractArray(abc.ABC):

    def __init__(self, num_images: int = None,
                 num_rows: int = None,
                 num_cols: int = None,
                 num_channels: int = None):
        if num_images and num_rows and num_cols and num_channels:
            self.DataArr = np.zeros(shape=(num_images, num_rows, num_cols,
                                           num_channels))

    def adjust_exposure(self, **kwargs):
        for img_num in range(self.DataArr.shape[0]):
            img_3d = self.DataArr[img_num, ...]
            for channel in range(img_3d.shape[-1]):
                img = img_3d[..., channel]
                



class DeepImgArray(AbstractArray):
    pass

    def adjust_gamma(self):
        self.adjust_exposure()

