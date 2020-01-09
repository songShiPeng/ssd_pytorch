import cv2 as cv
import os.path as osp
img=cv.imread('D:\data\study\mlData\\bigWork\core_500\Image\core_battery00000031xz90.jpg')
line = open('D:\data\study\mlData\\bigWork\core_500\Annotation\core_battery00000031xz90.txt', "r", encoding='utf-8').readline()
strs = line.split(' ')
cv.rectangle(img, (int(strs[2]), int(strs[3])), (int(strs[4]), int(strs[5])), (255, 255, 0), 2)
# cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
cv.namedWindow('input_image',0)
cv.resizeWindow('input_image',img.shape[1],img.shape[0])
cv.imshow('input_image', img)
cv.waitKey(0)
cv.destroyAllWindows()