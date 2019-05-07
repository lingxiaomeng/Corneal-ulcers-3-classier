import argparse

parser = argparse.ArgumentParser(description='The option of IQA Processing')

parser.add_argument('--GPU', type=str, default='0', help='the CUDA device you will use')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Only Support DIRMDB firstly
parser.add_argument('--prepare', action='store_false', help='Prepare Dataset')
parser.add_argument('--data_dir', type=str, default='D:\Projects\jiaomo-3classier\Image',
                    help='data directory')
parser.add_argument('--class1', type=str, default='/点染/', help='class1 directory label=0')
parser.add_argument('--class2', type=str, default='/点片混合/', help='class2 directory label=1')
parser.add_argument('--class3', type=str, default='/片染/', help='class3 directory label=2')

parser.add_argument('--filetype', type=str, default='jpg', help='dataset type')

parser.add_argument('--h5dir', type=str, default='/hdf5_0.7_299x299_5fold', help='hdf5储存位置')
parser.add_argument('--height', type=int, default=299, help='image height')
parser.add_argument('--width', type=int, default=299, help='image width')
parser.add_argument('--save', type=str, default='D:\Projects\jiaomo-3classier\model/model_inception_v3/',
                    help='trained model to save')

parser.add_argument('--train_percent', type=int, default=0.3, help='测试集+验证集占比')
parser.add_argument('--val_number', type=int, default=0, help='验证集数目')

parser.add_argument('--model', type=str, default='inceptionv3', help='training model')
parser.add_argument('--pre_train', type=str, default='../model/pre_train/inception_v3/inception_v3.ckpt',
                    help='the pre trained model directory')

parser.add_argument('--n_color', type=int, default=3, help='the channels used in training')
parser.add_argument('--cross_validation', type=int, default=0, help='the root for cross validation')
parser.add_argument('--class_num', type=int, default=2, help='the class_num for IQA')

# Training option

parser.add_argument('--early_stopping', type=int, default=1000, help='patience')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--online', type=bool, default=False, help='training online')
parser.add_argument('--batch_size', type=int, default=8, help='')  # Res_net:13 #mobile nas:13 //inception v3 8
parser.add_argument('--epoch', type=int, default=1000, help='')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
