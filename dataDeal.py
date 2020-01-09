import cv2 as cv
import random
import os
import numpy as np
import Config
import os.path as osp
# 添加椒盐噪声，prob:噪声比例 
def sp_noiseImg(image,prob):
  output = np.zeros(image.shape,np.uint8)
  thres = 1 - prob 
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      rdn = random.random()
      if rdn < prob:
        output[i][j] = 0
      elif rdn > thres:
        output[i][j] = 255
      else:
        output[i][j] = image[i][j]
  return output

# 添加高斯噪声
# mean : 均值
# var : 方差
def gasuss_noiseImg(image, mean=0, var=0.01):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


# 旋转图像，输入文件名、输出文件名，旋转角度
def rotationImg(img_file1, ra):
    # 获取图片尺寸并计算图片中心点
    (h, w) = img_file1.shape[:2]
    center = (w / 2, h / 2)

    M = cv.getRotationMatrix2D(center, ra, 1.0)

    # print(M[0])
    # M + = 0.01
    if (ra % 180 == 0):
        rotated = cv.warpAffine(img_file1, M, (w, h))
    else:
        M[0][2] += (h - w)/2
        M[1][2] += (w - h)/2
        rotated = cv.warpAffine(img_file1, M, (h, w))
    # cv.imshow("rotated", rotated)
    # cv.waitKey(0)
    return rotated,M
def write_txt(path,text):
    f = open(path,'w',encoding='utf-8')
    f.write(text)
    f.close()
for root,dir,text_paths in os.walk(Config.dataset_root + '/core_500/Annotation'):
    for text_path in text_paths:
        if('gs' in text_path or 'jy' in text_path or 'xz' in text_path):
            continue
        print(text_path)
        file_name, ext = osp.splitext(text_path)
        try:
            file = open(root + '/' + text_path, "r", encoding='utf-8')
            line = file.readline()
            if('带电芯' not in line):
                for cc in range(10):
                    line=file.readline()
                    if('带电芯' in line or line == ''):
                        break
                if('带电芯' not in line):
                    continue
        except UnicodeDecodeError as e:
            print(text_path)
        sorce_path_pre = Config.dataset_root + '/core_500/Image/' + text_path.split('.')[0]
        text_path_pre =  Config.dataset_root + '/core_500/Annotation/' + text_path.split('.')[0]
        img=cv.imread(sorce_path_pre + ".jpg")

        strs = line.split()
        print(line)
        # 椒盐噪声0.01
        imgJy = sp_noiseImg(img,0.01)
        cv.imwrite(sorce_path_pre + 'jy001.jpg',imgJy)
        write_txt(text_path_pre + "jy001.txt",line)

        # 高斯001
        imgGs = gasuss_noiseImg(img,0,0.01)
        cv.imwrite(sorce_path_pre + 'gs001.jpg',imgGs)
        write_txt(text_path_pre + "gs001.txt",line)
        # 旋转180
        img_xz180,M = rotationImg(img,180)
        # cv.rectangle(img180, (int(strs[2]), int(strs[3])), (int(strs[4]), int(strs[5])), (255, 255, 0), 2)
        begin = np.dot(M,np.array([[int(strs[2])],[int(strs[3])],[1]]))
        end = np.dot(M,np.array([[int(strs[4])],[int(strs[5])],[1]]))
        # cv.rectangle(img_xz180, (begin[0],begin[1]), (end[0],end[1]), (255, 255, 0), 2)
        line_xz180 = strs[0] + ' ' + strs[1] + ' ' + str(int(begin[0])) + ' ' + str(int((begin[1]))) + ' ' + str(int((end[0]))) + ' ' + str(int((end[1])))
        # print(line_xz180)
        cv.imwrite(sorce_path_pre + 'xz180.jpg',img_xz180)
        write_txt(text_path_pre + "xz180.txt",line_xz180)
        # 椒盐噪声0.02
        imgJy = sp_noiseImg(img_xz180, 0.01)
        cv.imwrite(sorce_path_pre + 'jy002.jpg', imgJy)
        write_txt(text_path_pre + "jy002.txt", line_xz180)

        # 高斯002
        imgGs = gasuss_noiseImg(img_xz180, 0, 0.01)
        cv.imwrite(sorce_path_pre + 'gs002.jpg', imgGs)
        write_txt(text_path_pre + "gs002.txt", line_xz180)
        # 旋转90
        img_xz90,M = rotationImg(img,90)
        # cv.rectangle(img180, (int(strs[2]), int(strs[3])), (int(strs[4]), int(strs[5])), (255, 255, 0), 2)
        begin = np.dot(M,np.array([[int(strs[2])],[int(strs[3])],[1]]))
        end = np.dot(M,np.array([[int(strs[4])],[int(strs[5])],[1]]))
        # cv.rectangle(img_xz180, (begin[0],begin[1]), (end[0],end[1]), (255, 255, 0), 2)
        line_xz90 = strs[0] + ' ' + strs[1] + ' ' + str(int(begin[0])) + ' ' + str(int((begin[1]))) + ' ' + str(int((end[0]))) + ' ' + str(int((end[1])))
        cv.imwrite(sorce_path_pre + 'xz90.jpg',img_xz90)
        write_txt(text_path_pre + "xz90.txt",line_xz90)
        # 椒盐噪声0.03
        imgJy = sp_noiseImg(img_xz90, 0.01)
        cv.imwrite(sorce_path_pre + 'jy003.jpg', imgJy)
        write_txt(text_path_pre + "jy003.txt", line_xz90)
        # 高斯003
        imgGs = gasuss_noiseImg(img_xz90, 0, 0.01)
        cv.imwrite(sorce_path_pre + 'gs003.jpg', imgGs)
        write_txt(text_path_pre + "gs003.txt", line_xz90)
        # 旋转270
        img_xz270,M = rotationImg(img,270)
        # cv.rectangle(img180, (int(strs[2]), int(strs[3])), (int(strs[4]), int(strs[5])), (255, 255, 0), 2)
        begin = np.dot(M,np.array([[int(strs[2])],[int(strs[3])],[1]]))
        end = np.dot(M,np.array([[int(strs[4])],[int(strs[5])],[1]]))
        # cv.rectangle(img_xz270, (begin[0],begin[1]), (end[0],end[1]), (255, 255, 0), 2)
        line_xz270 = strs[0] + ' ' + strs[1] + ' ' + str(int(begin[0])) + ' ' + str(int((begin[1]))) + ' ' + str(int((end[0]))) + ' ' + str(int((end[1])))
        cv.imwrite(sorce_path_pre + 'xz270.jpg',img_xz270)
        write_txt(text_path_pre + "xz270.txt",line_xz270)

        # cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
        # cv.imshow('input_image', img_xz180)
        # cv.waitKey(0)
        # cv.destroyAllWindows()