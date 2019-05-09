import cv2
import numpy as np

from keras.utils import to_categorical
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
    # print(train_labels)
    # for i in range(len(train_labels)):
    #     if train_labels[i] == 2:
    #         train_labels[i] = 0

    # print(train_labels)

    new_train_imgs = np.empty((241, 299, 299, 3))
    print(len(new_train_imgs))
    print(new_train_imgs.shape)
    new_train_labels = []
    idx = 0
    for i in range(len(train_labels)):
        if train_labels[i] == 2:
            img = plt.imread(train_file[i])
            img = cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)
            new_train_imgs[idx] = img
            new_train_labels.append(1)
            idx += 1
        elif train_labels[i] == 1:
            img = plt.imread(train_file[i])
            img = cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)
            new_train_imgs[idx] = img
            new_train_labels.append(0)
            idx += 1


    print(len(new_train_labels))
    one_hot_train = to_categorical(np.array(train_labels))
    new_one_hot_train = to_categorical(np.array(new_train_labels))
    if len(val_labels > 0):
        one_hot_val = to_categorical(np.array(val_labels))
    #    onehot_val = to_categorical(data1)
    # print(onehot_train)
    # print(onehot_val)
    model = Model_inception_v3(args)
    model.train(new_train_imgs, new_one_hot_train, val_imgs, None)
