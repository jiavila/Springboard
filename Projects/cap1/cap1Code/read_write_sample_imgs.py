import numpy as np
import matplotlib.pyplot as plt
from os import path
import os


def open_cv10_data():
    path_cv10_data = path.abspath('D:\Documents\Springboard\ProjectData\ddsm-mammography\cv10_data')
    data_cv10 = np.load(path_cv10_data + '\\cv10_data.npy')
    labels_cv10 = np.load(os.path.join(path_cv10_data, '../cv10_labels.npy'))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print('Dir path: ' + dir_path)

    path_sample_imgs = os.path.join(dir_path, '../sample_imgs')

    return [path_cv10_data, data_cv10, labels_cv10, path_sample_imgs]


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
