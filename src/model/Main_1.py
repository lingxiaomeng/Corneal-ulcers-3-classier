import cv2
import numpy as np

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from data import DataLoader
import utils
from Model_NAS import Model_Nas
from Model_inceptionv3 import Model_inception_v3
from Model_Res import Model_resNet
from option import args
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_loader = DataLoader(args)
    train_imgs, train_labels, train_file = data_loader.get_train()
    val_imgs, val_labels, val_file = data_loader.get_val()
    train_imgs = utils.dataset_normalized(train_imgs)
    val_imgs = utils.dataset_normalized(val_imgs)
    print(train_labels)
    for i in range(len(train_labels)):
        if train_labels[i] == 2:
            train_labels[i] = 0

    print(train_labels)

    x_train, x_test, y_train, y_test = train_test_split(train_imgs, train_labels,
                                                        test_size=0.35,
                                                        random_state=1)
    if len(val_labels > 0):
        one_hot_val = to_categorical(np.array(val_labels))
    #    onehot_val = to_categorical(data1)
    # print(onehot_train)
    # print(onehot_val)
    model = Model_inception_v3(args)
    model.train(x_train, to_categorical(np.array(y_train)), x_test, to_categorical(np.array(y_test)))
    # model.train_5fold(train_imgs, train_labels)
