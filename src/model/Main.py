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
    train_imgs = utils.dataset_normalized(train_imgs)
    x0 = 0
    x1 = 0
    x2 = 0

    for l in train_labels:
        if l == 0:
            x0 += 1
        elif l == 1:
            x1 += 1
        elif l == 2:
            x2 += 1
    print('{} {} {}'.format(x0, x1, x2))
    new_train_imgs = np.empty((x1 + 3 * x2, 299, 299, 3))
    print(len(new_train_imgs))
    print(new_train_imgs.shape)
    new_train_labels = []
    idx = 0
    num = 0
    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            new_train_imgs[idx] = train_imgs[i]
            new_train_labels.append(0)
            idx += 1
        elif train_labels[i] == 2:
            new_train_imgs[idx] = train_imgs[i]
            new_train_imgs[x1 + x2 + num] = train_imgs[i]
            new_train_imgs[x1 + 2 * x2 + num] = train_imgs[i]
            new_train_labels.append(1)
            idx += 1
            num += 1

    for i in range(2*x2):
        new_train_labels.append(1)

    print(num)
    print(new_train_labels)
    print(len(new_train_labels))
    print(len(new_train_imgs))
    x_train, x_test, y_train, y_test = train_test_split(new_train_imgs, new_train_labels,
                                                        test_size=0.4,
                                                        random_state=1)

    print(y_train)
    print(len(y_train))
    new_one_hot_train = to_categorical(np.array(y_train))
    new_one_bot_val = to_categorical(np.array(y_test))

    # print(new_one_hot_train)
    #    onehot_val = to_categorical(data1)
    # print(onehot_train)
    # print(onehot_val)
    model = Model_inception_v3(args)
    model.train(x_train, new_one_hot_train, x_test, new_one_bot_val)
