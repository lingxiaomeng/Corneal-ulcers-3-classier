import glob
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:
        return f["image"][()]


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset('image', data=arr, dtype=arr.dtype)


class DataLoader:
    def __init__(self, args):
        self.hdf5_dir = args.data_dir + args.h5dir
        self.c = args.n_color
        self.seed = args.seed
        self.cross = args.cross_validation
        self.type = args.filetype
        self.class1_dir = args.data_dir + args.class1
        self.class2_dir = args.data_dir + args.class2
        self.class3_dir = args.data_dir + args.class3
        self.height = args.height
        self.width = args.width
        self.train_percent = args.train_percent
        self.val_number = args.val_number

    def _access_dataset(self):
        class1_list = glob.glob(self.class1_dir + '*.' + self.type)
        class2_list = glob.glob(self.class2_dir + '*.' + self.type)
        class3_list = glob.glob(self.class3_dir + '*.' + self.type)
        p1 = len(class1_list)
        p2 = len(class2_list)
        p3 = len(class3_list)
        print("{} {} {}".format(p1, p2, p3))
        data_list = class1_list + class2_list + class3_list
        labels = []
        filenames = []
        imgs = np.empty((len(data_list), self.height, self.width, self.c))

        for idx in tqdm(range(len(data_list))):
            file = data_list[idx]
            img = plt.imread(file)
            # img = img[20:, :, :]
            if img.shape[0] != self.height:
                img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_CUBIC)
            if self.c != 3:  # TODO
                raise NotImplementedError('The one channels training hasn\'t been implemented')
            else:
                imgs[idx] = img

            if idx < p1:
                labels.append(0)
            elif idx < p1 + p2:
                labels.append(1)
            elif idx < p1 + p2 + p3:
                labels.append(2)
            filenames.append(str(file))
        assert len(data_list) == len(labels)
        return imgs, np.asarray(labels), np.asarray(filenames)

    def prepare_dataset(self):
        print('[Prepare Data]')
        imgs, labels, filenames = self._access_dataset()
        if self.cross != 0:  # Use Cross Validation
            # TODO
            if os.path.exists(self.data_dir + '/cv.txt'):
                pass  # TODO
        else:
            x_train, x_test, y_train, y_test, name_train, name_test = train_test_split(imgs, labels, filenames,
                                                                                       test_size=self.train_percent,
                                                                                       random_state=self.seed)
        print('Training : {} images'.format(len(y_train)))
        print('Testing: {} images'.format(len(y_test)))
        if not os.path.exists(self.hdf5_dir + '/train'):
            os.makedirs(self.hdf5_dir + '/train')
        write_hdf5(x_train, self.hdf5_dir + '/train/train.hdf5')

        np.savetxt(self.hdf5_dir + '/train/train_labels.txt', y_train.astype(np.int64))
        np.savetxt(self.hdf5_dir + '/train/train_filename.txt', name_train, fmt='%s', encoding='utf-8')

        if not os.path.exists(self.hdf5_dir + '/test'):
            os.makedirs(self.hdf5_dir + '/test')
        write_hdf5(x_test, self.hdf5_dir + '/test/test.hdf5')
        np.savetxt(self.hdf5_dir + '/test/test_labels.txt', y_test.astype(np.int64))
        np.savetxt(self.hdf5_dir + '/test/test_filename.txt', name_test, fmt='%s', encoding='utf-8')

        print('[Finish]')

    def get_train(self):
        imgs_train = load_hdf5(self.hdf5_dir + '/train/train.hdf5')
        labels_train = np.loadtxt(self.hdf5_dir + '/train/train_labels.txt')
        filename_train = np.loadtxt(self.hdf5_dir + '/train/train_filename.txt', dtype=str, encoding='utf-8')

        return imgs_train, labels_train, filename_train

    def get_val(self):
        imgs_val = load_hdf5(self.hdf5_dir + '/test/test.hdf5')
        labels_val = np.loadtxt(self.hdf5_dir + '/test/test_labels.txt')
        # return imgs_va[:15], labels_val[:15]
        filename_val = np.loadtxt(self.hdf5_dir + '/test/test_filename.txt', dtype=str, encoding='utf-8')
        return imgs_val[:self.val_number], labels_val[:self.val_number], filename_val[:self.val_number]

    def get_test(self):
        imgs_test = load_hdf5(self.hdf5_dir + '/test/test.hdf5')
        labels_test = np.loadtxt(self.hdf5_dir + '/test/test_labels.txt')
        filename_test = np.loadtxt(self.hdf5_dir + '/test/test_filename.txt', dtype=str, encoding='utf-8')
        return imgs_test[self.val_number:], labels_test[self.val_number:], filename_test[self.val_number:]
