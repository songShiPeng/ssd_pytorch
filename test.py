import torch
import ml_data
from torch.autograd import Variable
from detection import *
from ssd_net_vgg import *
import torch.nn as nn
import numpy as np
import cv2
import utils
import Config
from PIL import Image
import os

# os.environ['CUDA_VISIBLE_DEVICES']='0'
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
predict_file_root = os.path.abspath('.') + "/predicted_file/"
image_path_root = "D:/data/study/mlData/testscript/Image_test"
core_file = open(predict_file_root + "det_test_带电芯充电宝.txt", 'w')
core_less_file = open(predict_file_root + "det_test_不带电芯充电宝.txt", 'w')
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229), (158, 218, 229), (158, 218, 229)]

net = SSD()  # initialize SSD
net = torch.nn.DataParallel(net)
net = net.cuda()
net.train(mode=False)
net.load_state_dict(torch.load('./weights/ssd300_VOC_100000.pth', map_location=lambda storage, loc: storage))
for image_path in os.listdir(image_path_root):
    # for image_path in files:
    image_full_path = image_path_root + "/" + image_path
    image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
    # image = cv2.imread("./testml.jpg", cv2.IMREAD_COLOR)
    # x = cv2.resize(image, (300, 300)).astype(np.float32)
    width = image.shape[0]
    hight = image.shape[1]
    # x = cv2.resize(image, (int(300*width/hight), 300)).astype(np.float32)
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    # x = image.astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    # plt.imshow(x)
    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
    # if torch.cuda.is_available():
    if 0 == 1:
        xx = xx.cuda()
    y = net(xx)
    softmax = nn.Softmax(dim=-1)
    detect = Detect(Config.class_num, 0, 200, 0.01, 0.45)
    priors = utils.default_prior_box()

    loc, conf = y
    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    detections = detect(
        loc.view(loc.size(0), -1, 4),
        softmax(conf.view(conf.size(0), -1, Config.class_num)),
        torch.cat([o.view(-1, 4) for o in priors], 0)
    ).data

    labels = ml_data.SIXray_CLASSES
    top_k = 10

    # plt.imshow(rgb_image)  # plot the image for matplotlib

    # scale each detection back up to the image
    scale = torch.Tensor(image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.01:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]

            display_txt = '%.2f' % (score)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            image_name = image_path[0:image_path.find('.')]
            if label_name == 'core':
            # label_name = '带电芯充电宝'
                core_file.write(image_name + " " + display_txt + " " + str(pt[0]) + " " + str(pt[1])+ " " + str(pt[2]) + " " + str(pt[3]) + "\n")
            elif label_name == 'core_less':
                core_less_file.write(image_name + " " + display_txt + " " + str(pt[0]) + " " + str(pt[1])+ " " + str(pt[2]) + " " + str(pt[3]) + "\n")
            # coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            # color = colors_tableau[i]
            # cv2.rectangle(image, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)
            # cv2.putText(image, display_txt, (int(pt[0]), int(pt[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1,
            #             8)
            j += 1
    # cv2.imshow('test', image)
    # cv2.waitKey(100000)
    # break
