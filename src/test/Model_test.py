import matplotlib
import numpy as np
from keras.applications.nasnet import NASNetMobile
from keras.utils import to_categorical
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
model = load_model(model_inception)
model.evaluate(x, to_categorical(np.array(x_label)))
