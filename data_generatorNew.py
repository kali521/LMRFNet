import h5py
import glob
import cv2
import numpy as np
import scipy
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch
import scipy.io as sio
from scipy.io import savemat,loadmat

# patch_size, stride = 100, 40
# patch_size1, stride1 = 100, 40

patch_size, stride = 80, 20
patch_size1, stride1 =80, 20
aug_times = 1 
scales = [1]  
batch_size = 64


class DenoisingDataset(Dataset):

    def __init__(self, xs, sigma): #Initializes the state of a newly created object; called immediately after an object is created
        super(DenoisingDataset, self).__init__()
        #self.label = label
        self.xs = xs  # Clean image
        self.sigma = sigma  # Noise


    def __getitem__(self, index): #Index a sample in the dataset
        #batch_z = self.label[index]
        batch_x = self.xs[index]
        noise = self.sigma[index]
        
        batch_y= batch_x + noise


        # noise = torch.randn(batch_x.size()).mul_(self.sigma)

        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)  # batchsize


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patcheszaosheng(file_name):
    
    img1=scipy.io.loadmat(file_name)
    img = img1['block_data'][:]
    # img = img1['a'][:]
    # img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(img, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled - patch_size1 + 1, stride1):
            for j in range(0, w_scaled - patch_size1 + 1, stride1):
                x = img_scaled[i:i + patch_size1, j:j + patch_size1]
                #x = x/x.max()
                #aa = 2 * (np.random.rand())
                x = (3.6 * (np.random.rand())) * x
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datageneratorzaosheng(data1_dir='', verbose=False):
    # generate noise patches from a dataset
    file_list = glob.glob(data1_dir + '/*.mat')  # get name list of all .png files
    # initrialize
    data1 = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patcheszaosheng(file_list[i]) #gen_patchszaosheng(filename)

        for patch in patches:
            data1.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    data1 = np.array(data1, dtype='float32')
    data1 = np.expand_dims(data1, axis=3)  
    discard_n = len(data1) - len(data1) // batch_size * batch_size  # because of batch namalization
    data1 = np.delete(data1, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data1


def gen_patches(file_name):
  
    img1=scipy.io.loadmat(file_name)

    img = img1['block_data'][:]  ##MAT file variable name
    # img = img1['a'][:]
    # img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)  # s=scales=1
        img_scaled = cv2.resize(img, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datagenerator(data_dir='', verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir + '/*.mat')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:
            data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='float32')
    data = np.expand_dims(data, axis=3)
    discard_n = len(data) - len(data) // batch_size * batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data

if __name__ == '__main__':
    
    data1 = datageneratorzaosheng(data1_dir='/home/zhangzeyuan1/d2sm-master/ceshiwenjian/noisepatch')
    data = datagenerator(data_dir='/home/zhangzeyuan1/d2sm-master/ceshiwenjian/cleanpatch')
   