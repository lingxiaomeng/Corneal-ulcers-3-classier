import numpy as np

from keras.utils import to_categorical
from data import DataLoader
import utils
from Model_NAS import Model_Nas
from Model_inceptionv3 import Model_inception_v3
from Model_Res import Model_resNet
from option import args

if __name__ == '__main__':
    data_loader = DataLoader(args)
    train_imgs, train_labels, train_file = data_loader.get_train()
    val_imgs, val_labels, val_file = data_loader.get_val()
    train_imgs = utils.dataset_normalized(train_imgs)
    val_imgs = utils.dataset_normalized(val_imgs)

    one_hot_train = to_categorical(np.array(train_labels))
    # one_hot_val = to_categorical(np.array(val_labels))
    #    onehot_val = to_categorical(data1)
    # print(onehot_train)
    # print(onehot_val)
    model = Model_inception_v3(args)
    model.train_5fold(train_imgs, train_labels)
