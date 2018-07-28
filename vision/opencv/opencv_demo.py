import cv2
import os

data_src_path = '../Keras-image-classifer-framework-master/data/'

# img = cv2.imread(os.path.join(data_src_path, 'traffic-sign/train/00000/01153_00000.png'))
# img = cv2.resize(img, (100,300))
# cv2.imshow('origin', img)
# cv2.imwrite('01153_00000_big.png', img)
# cv2.waitKey(0)

img = cv2.imread(os.path.join(data_src_path, 'weather-sign/train/00002/20180728175815.png'))
img = cv2.resize(img, (100,300))
cv2.imshow('origin', img)
cv2.imwrite('QQ截图20180728175815.png', img)
cv2.waitKey(0)
