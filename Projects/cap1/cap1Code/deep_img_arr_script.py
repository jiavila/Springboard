from skimage import data
import numpy as np
import matplotlib.pyplot as plt

from deepimgarr import DeepArrayExposure


astronaut = data.astronaut()\

astronauts = np.empty(shape=(2,) + astronaut.shape)

astronauts[0, ...] = astronaut
astronauts[1, ...] = astronaut

dax = DeepArrayExposure(images=astronauts)



my_test_arr = DeepArrayExposure(2, 3, 3, 1)

print("debug holder")