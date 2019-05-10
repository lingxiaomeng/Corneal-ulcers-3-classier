import matplotlib
import numpy as np
from keras.applications import InceptionV3
from keras.applications.nasnet import NASNetMobile
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.python.keras.models import load_model

import utils
from Model_inceptionv3 import focal_loss
from data import DataLoader
from model_v2 import add_new_last_layer
from option import args

model_Res = 'D:\Projects\jiaomo-3classier\model\model_resnet\ResNet_best_weights.h5'
mode_fold_res_5 = 'D:\Projects\jiaomo-master\Model\model5_resNet5fold\ResNet_best_weights_fold_4.h5'
model_inception = 'D:\Projects\jiaomo-3classier\model\model_inception_v3_02_1\Inception_v3_best_weights.h5'
model_inception5fold = 'D:\Projects\jiaomo-master\Model\model_inception_v35fold\Inception_v3_best_weights_fold_4.h5'
data_loader = DataLoader(args)
x, x_label, x_file = data_loader.get_test()

x = utils.dataset_normalized(x)
matplotlib.use('Agg')
model_021 = load_model(model_inception)

y_other = model_021.predict(x)

model_origin = load_model("D:\Projects\jiaomo-master\Model\model_inception_v3\inception_v3_best_weights.h5")

TP = 0
TN = 0
FP = 0
FN = 0

# for i in range(len(y_other)):
#     if x_label[i] == 0 or x_label[i] == 2:
#         y_other[i][0]>

y_original = model_origin.predict(x)

y_label = np.empty((len(y_original)))

print(y_other)
print(y_original)

for i in range(len(y_original)):
    if y_original[i][0] > 0.5:
        y_label[i] = 1
    else:
        y_label[i] = 0

for i in range(len(y_label)):
    if y_label[i] == 1:
        if y_other[i][0] > 0.6:
            y_label[i] = 2

print(y_label)
errorfile = []

x00 = 0
x01 = 0
x02 = 0
x10 = 0
x11 = 0
x12 = 0
x20 = 0
x21 = 0
x22 = 0


def indexmax(xx):
    index = 0
    max = 0
    i = 0
    for a in xx:
        if a > max:
            max = a
            index = i
        i += 1
    return index


i = 0
for d in y_label:
    if x_label[i] == 0:
        index = d
        if index == 0:
            x00 += 1
        if index == 1:
            x01 += 1
            errorfile.append(x_file[i])
        if index == 2:
            x02 += 1
            errorfile.append(x_file[i])
    if x_label[i] == 1:
        index = d
        if index == 0:
            x10 += 1
            errorfile.append(x_file[i])
        if index == 1:
            x11 += 1
        if index == 2:
            x12 += 1
            errorfile.append(x_file[i])

    if x_label[i] == 2:
        index = d
        if index == 0:
            x20 += 1
            errorfile.append(x_file[i])
        if index == 1:
            x21 += 1
            errorfile.append(x_file[i])
        if index == 2:
            x22 += 1
    i += 1

print("{} {} {}".format(x00, x01, x02))
print("{} {} {}".format(x10, x11, x12))
print("{} {} {}".format(x20, x21, x22))
