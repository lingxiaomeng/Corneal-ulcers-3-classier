

import os

from data import DataLoader
from option import args

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

if __name__ == '__main__':
    print('[Start]')
    data_loader = DataLoader(args)
    data_loader.prepare_dataset()
