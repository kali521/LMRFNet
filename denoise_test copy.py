import numpy as np
import argparse
import os, time, datetime
import matplotlib.pyplot as plt
from Cut_combine import cut, combine
import torch
from scipy.io import savemat
from scipy.io import loadmat
from tqdm import tqdm
# Load data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/1', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['guiyihua'], help='directory of test dataset')
    parser.add_argument('--model_dir', default=os.path.join('/home/zhangzeyuan1/d2sm-master/models', 'LMRFsfq_50'), help='directory of the model')
    parser.add_argument('--model_name', default='model_150.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seismic_noisy = loadmat('/home/zhangzeyuan1/d2sm-master/quzaoshiyan/Data1.mat')
seismic_noisy = seismic_noisy['data']  #test

seismic_height, seismic_width = seismic_noisy.shape

patch_size = 80
patches, strides_x, strides_y, fill_arr_h, fill_arr_w = cut(seismic_noisy, patch_size, patch_size, patch_size)
# Load model
model = torch.load(os.path.join(args.model_dir, args.model_name), weights_only=False)
# device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Predict on the sliced patches
predict_datas = []
for patch in patches:
    patch = np.array(patch)
    patch = patch.reshape(1, 1, patch.shape[0], patch.shape[1])
    patch = torch.from_numpy(patch)
    patch = patch.to(device=device, dtype=torch.float32)
    predict_data = model(patch)
    predict_data = predict_data.data.cpu().numpy()
    predict_data = predict_data.squeeze()
    predict_datas.append(predict_data)
seismic_predict = combine(predict_datas,  patch_size, strides_x, strides_y, seismic_height, seismic_width)

seismic_dict = {'denoised': seismic_predict}


savemat('/home/zhangzeyuan1/d2sm-master/quzaoshiyan/predict.mat', seismic_dict)


print("Prediction result saved as predict.mat file.")