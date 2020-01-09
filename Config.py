import os.path as osp
sk = [ 15, 30, 60, 111, 162, 213, 264 ]
feature_map = [ 38, 19, 10, 5, 3, 1 ]
steps = [ 8, 16, 32, 64, 100, 300 ]
image_size = 300
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
MEANS = (104, 117, 123)
batch_size = 1
data_load_number_worker = 0
lr = 1e-5
momentum = 0.9
weight_decacy = 5e-4
gamma = 2
# VOC_ROOT = osp.join('./', "VOCdevkit/")
VOC_ROOT = "D:/data/study/mlData/bigWork/"
dataset_root = VOC_ROOT
dataset_test_root = "D:/data/study/mlData/test/"
use_cuda = True
# lr_steps = (80000, 100000, 120000)
lr_steps = (20000, 60000, 100000)
# lr_steps = (200, 400, 800)
# max_iter = 120000
max_iter = 100000
class_num = 3
epoch_num = 200

# 测试文件路径
