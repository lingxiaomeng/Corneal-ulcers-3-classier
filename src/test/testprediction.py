import matplotlib
import numpy as np
from keras.applications.nasnet import NASNetMobile
from tensorflow.python.keras.models import load_model

import utils
from data import DataLoader
from option import args

# 忽略硬件加速的警告信息

model_nas = 'C:\\Users\mlx\PycharmProjects\角膜\Model\model3_nas\\NASNet_best_weights.h5'
model_Res = 'D:\Projects\jiaomo-master\Model\model5_resNet\ResNet_best_weights.h5'
mode_fold_res_5 = 'D:\Projects\jiaomo-master\Model\model5_resNet5fold\ResNet_best_weights_fold_4.h5'
model_inception = 'D:\Projects\jiaomo-master\Model\model_inception_v3\inception_v3_best_weights.h5'
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
model = load_model(model_inception5fold)
##################################################
# model = NASNetMobile(classes=2, include_top=True, weights=model_nas)
# model.load_weights(model_nas)
###################################
# model.summary()
y = model.predict(x)

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
