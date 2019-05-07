import shutil

import numpy as np
import os

labels = np.loadtxt('D:\Projects\jiaomo-master\Image\data1\\a.txt')
dir = 'D:\Projects\jiaomo-master\Image\data1\片染点片\\'
print(labels)

for img in labels:
    filename = dir + str(int(img[1])) + '.jpg'
    if img[0] == 1:
        shutil.copy(filename, "D:\Projects\jiaomo-master\Image\data1\点片混合\\")
    else:
        shutil.copy(filename, "D:\Projects\jiaomo-master\Image\data1\片染\\")
