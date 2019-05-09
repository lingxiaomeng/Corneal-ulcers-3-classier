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

value = 1 / 3

model_Res = 'D:\Projects\jiaomo-3classier\model\model_resnet\ResNet_best_weights.h5'
mode_fold_res_5 = 'D:\Projects\jiaomo-master\Model\model5_resNet5fold\ResNet_best_weights_fold_4.h5'
model_inception = 'D:\Projects\jiaomo-3classier\model\model_inception_v3_1_2\Inception_v3_best_weights.h5'
model_inception5fold = 'D:\Projects\jiaomo-master\Model\model_inception_v35fold\Inception_v3_best_weights_fold_4.h5'
data_loader = DataLoader(args)
x, x_label, x_file = data_loader.get_test()

x = utils.dataset_normalized(x)
matplotlib.use('Agg')
# base_model = NASNetMobile(weights=None, include_top=False)
# model = add_new_last_layer(base_model, 2)
# # model.compile(optimizer=Adam(lr=0.0001, beta_1=0.1),
# #                            loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.load_weights(model_nas)

###################################################
model = InceptionV3(include_top=False, weights='imagenet')
model = add_new_last_layer(model, nb_classes=2)
# self.model.compile(optimizer=Adam(lr=0.0001, beta_1=0.1),
#                    loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.compile(optimizer=Adam(lr=0.0001, beta_1=0.1),
              loss=[focal_loss(alpha=.25, gamma=2)], metrics=['categorical_accuracy'])

model.load_weights(model_inception)
##################################################
# model = NASNetMobile(classes=2, include_top=True, weights=model_nas)
# model.load_weights(model_nas)
###################################
# model.summary()
idx = 0
x_new_label = []
y = model.predict(x)
for i in range(len(x_label)):
    if x_label[i] == 2:
        x_new_label.append(1)
        idx += 1
    elif x_label[i] == 1:
        x_new_label.append(0)
        idx += 1
    else:
        x_new_label.append(0)
        idx += 1

model.evaluate(x, to_categorical(np.array(x_new_label)))
print(y)


def predict(y, num):
    errorfile = []
    goodnum = 0
    total = 0
    total_good = 0
    badnum = 0
    total_bad = 0
    i = 0
    for xx in y:
        if x_label[i] == 1:
            total_good += 1
            if xx[1] > num:
                goodnum += 1
            else:
                errorfile.append(x_file[i])
        if x_label[i] == 0:
            total_bad += 1
            if xx[0] >= 1 - num:
                badnum += 1
            else:
                errorfile.append(x_file[i])
        i += 1
        total += 1
    TP = badnum
    TN = goodnum
    FN = total_bad - badnum
    FP = total_good - goodnum
    Sen = TP / (TP + FN)
    spe = TN / (TN + FP)
    bacc = (Sen + spe) / 2
    ACC = (TP + TN) / (TP + TN + FP + FN)
    print('num={} Sensitivity={} Specificity={} Balanced Accuracy={} Accuracy={}'.format(num, Sen, spe, bacc, ACC))
    return Sen, spe, bacc, ACC, np.asarray(errorfile)


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
for d in y:
    if x_label[i] == 0:
        index = indexmax(d)
        if index == 0:
            x00 += 1
        if index == 1:
            x01 += 1
            errorfile.append(x_file[i])
        if index == 2:
            x02 += 1
            errorfile.append(x_file[i])
    if x_label[i] == 1:
        index = indexmax(d)
        if index == 0:
            x10 += 1
            errorfile.append(x_file[i])
        if index == 1:
            x11 += 1
        if index == 2:
            x12 += 1
            errorfile.append(x_file[i])

    if x_label[i] == 2:
        index = indexmax(d)
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

tt = 0
sen = []
spe = []
bacc = []
ACC = []
errf = []
for i in range(0, 101):
    a, b, c, d, e = predict(y, tt)
    tt += 0.01
    sen.append(a)
    spe.append(b)
    bacc.append(c)
    ACC.append(d)
    errf.append(e)

print(sen)
print(spe)
print(bacc)
print(ACC)
spe_1 = [1 - i for i in spe]
auc = 0
tt = 0
for i in range(0, len(spe_1) - 1):
    auc += (spe_1[i + 1] - spe_1[i]) * (sen[i] + sen[i + 1]) / 2

print('AUC={}'.format(auc))

index = ACC.index(max(ACC))
print('specificity={}'.format(spe[index]))
print('sensitivity={}'.format(sen[index]))
print('Best B-Accuracy={}'.format(bacc[index]))
print('Best acc{}'.format(ACC[index]))
np.savetxt(args.save + 'errorfile', errf[index], fmt='%s', encoding='utf-8')

print(errf[index])
