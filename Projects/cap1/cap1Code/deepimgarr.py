"""
DeepImgArray
------------

A class for applying manipulations to deep learning arrays.
"""

import numpy as np
import abc
from skimage import exposure as xp


class DeepArrayBase(abc.ABC):

    def __init__(self, images: np.ndarray):
        self.images = self.check_array_shape(images=images)

    @staticmethod
    def check_array_shape(images):
        if len(images.shape) != 4:
            raise TypeError(
                "images must be a numpy array with 4 dimensions, "
                "not {}".format(len(images.shape)))

        return images


    '''
    num_images: int = None,
                 num_rows: int = None,
                 num_cols: int = None,
                 num_channels: int = None):
        if num_images and num_rows and num_cols and num_channels:
            self.DataArr = np.zeros(shape=(num_images, num_rows, num_cols,
                                           num_channels)
    '''


class DeepArrayExposure(DeepArrayBase):

    def __init__(self, images):
        super().__init__(images=images)

    def adjust_exposure(self, exposure_method, **kwargs):
        for img_num in range(self.DataArr.shape[0]):
            img_3d = self.DataArr[img_num, ...]
            for channel in range(img_3d.shape[-1]):
                img = img_3d[..., channel]
                img_adj = exposure_method(img, kwargs)
                img_3d[..., channel] = img_adj

            # Store our adjusted image in our data array
            self.DataArr[img_num, ...] = img_3d

    def adjust_gamma(self, gamma, gain):
        self.adjust_exposure(exposure_method=xp.adjust_gamma,
                             gamma=gamma, gain=gain)


def main():
    pass


if __name__ == "__main__":
    main()
