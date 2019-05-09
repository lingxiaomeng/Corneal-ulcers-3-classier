import cv2
import matplotlib.pyplot as plt

img = plt.imread('D:\Projects\jiaomo-3classier\Image\点染\\1.jpg')
cv2.imwrite(img=img, filename='bef.jpg')
img = img[20:, :, :]


print(len(img[20:, :, ]))
# print(img)
print(len(img))
print(len(img[0]))
print(len(img[0][0]))
# cv2.imshow('a', img[:, :, ])
cv2.waitKey(0)
cv2.destroyAllWindows()
