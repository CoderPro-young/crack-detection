import cv2
# import matplotlib.pyplot as plt

path = './data/Cam4-2020_06_08_03_37_18_974-75-3700.jpg'

img = cv2.imread(path)
print(img.shape)
# cv2.imshow('window', img)
# k = cv2.waitKey(0)
# if(k == ord('s')):
#     cv2.imwrite('lena_.jpg', img)