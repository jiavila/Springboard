import numpy as np
import matplotlib.pyplot as plt
from os import path
import os
import deepimgbuilder

# run this only if the file isn't imported
if __name__ == '__main__':

    deep_image_builder_ex = deepimgbuilder()
    deep_image_builder_ex.set_path_data()
    deep_image_builder_ex.get_data()

    [path_cv10_data, data_cv10, labels_cv10, path_sample_imgs] = open_cv10_data()

    for indx, img in enumerate(data_cv10[0:5, :, :, 0]):
        fig = plt.figure(num=indx+1)
        print('im' + str(indx+1) + ': ' + str(img.shape))
        print('Type' + str(type(img)))
        plt.imshow(img, cmap='gray')
        plt.title('Sample Image ' + str(indx+1) + ', Label: ' + str(labels_cv10[indx]))
        plt.savefig(path_sample_imgs + '/im' +str(indx+1) + '.png')


    # plt.show('im' + str(indx+1) + '') # Uncomment to display
    print('Saved sample images in ' + path.abspath(path_sample_imgs))
