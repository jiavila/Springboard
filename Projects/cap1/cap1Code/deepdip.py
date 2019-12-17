"""
Simple digital image processing techniques applied to DeepImageBuilder
objects (i.e., images in a 4D numpy array)
"""
from skimage import exposure as xp
from deepimgbuilder import DeepImageBuilder

class DeepDip():

    def __init__(self, deep_image_builder=None):
        if type(deep_image_builder) is DeepImageBuilder:
            self.Dib = deep_image_builder
