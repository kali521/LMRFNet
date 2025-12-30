import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generatorNew as dg
from data_generatorNew import DenoisingDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
# from lossnettt190AB import BCDNet, Dice_Loss, count_param
import matplotlib.pyplot as plt
from LMRFNetsfq import LMRFNet
from sklearn.model_selection import train_test_split  

from tensorboardX import SummaryWriter
writer = SummaryWriter('logs')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Params  
parser = argparse.ArgumentParser(description='PyTorch LMRFNET')
parser.add_argument('--model', default='LMRFNET', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_data', default='/home/zhangzeyuan1/d2sm-master/data/DAScleanpatch_zzy', type=str, help='path of train data')
parser.add_argument('--zaosheng_data', default='/home/zhangzeyuan1/d2sm-master/data/DASnoisepatch_zzy', type=str, help='path of zaosheng data')
parser.add_argument('--n_epoch', default=100, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.n_epoch

save_dir = os.path.join('/home/zhangzeyuan1/d2sm-master/ceshiwenjian/model', args.model + '_' + '03')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    print('===> Building model')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    #model = UNet(n_channels=1,n_classes=1)
    model=LMRFNet()
    model.to(device)

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    model.train()
    criterion = nn.MSELoss(reduction='mean', size_average=True)  # PyTorch 0.4.1
    # criterion = nn.L1Loss()
    # criterion = Dice_Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.5)  # learning rates
    scheduler = CosineAnnealingLR(optimizer, args.n_epoch)
    
    # =================== Load and split the dataset ===================
    log('Loading clean and noisy data...')
    xs = dg.datagenerator(data_dir=args.train_data)
    sigma = dg.datageneratorzaosheng(data1_dir=args.zaosheng_data)

 
    min_len = min(len(xs), len(sigma))
    xs = xs[:min_len]
    sigma = sigma[:min_len]

    #  (80% train, 20% val)
    xs_train, xs_val, sigma_train, sigma_val = train_test_split(
        xs, sigma,
        test_size=0.2,
        random_state=42 
    )

    log(f'Train set size: {len(xs_train)}, Validation set size: {len(xs_val)}')

    #  tensor (NHWC -> NCHW)
    xs_train = torch.from_numpy(xs_train.transpose((0, 3, 1, 2))).float()
    sigma_train = torch.from_numpy(sigma_train.transpose((0, 3, 1, 2))).float()
    xs_val = torch.from_numpy(xs_val.transpose((0, 3, 1, 2))).float()
    sigma_val = torch.from_numpy(sigma_val.transpose((0, 3, 1, 2))).float()


    train_dataset = DenoisingDataset(xs_train, sigma_train)
    val_dataset = DenoisingDataset(xs_val, sigma_val)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    # =======================================================


    ##Plotting the loss curve, initializing the training loss history
    train_loss_history = []
    val_loss_history = []
    ##


    filename = 'LMRFNet.txt' #Total value for each epoch
    with open(filename, 'w') as f:
        best_val_loss = float('inf')  # Used to save the best model

        total_start_time = time.time()  

        for epoch in range(initial_epoch, n_epoch):
            # ------------------ Training phase ------------------
            model.train()
            epoch_loss = 0
            start_time = time.time()

            for n_count, batch_yx in enumerate(train_loader):
                optimizer.zero_grad()
                if cuda:
                    batch_x, batch_y = batch_yx[1].to(device), batch_yx[0].to(device)

                outsx1 = model(batch_y) 
                loss = criterion(outsx1, batch_x)  

                epoch_loss += loss.item()
                loss.backward()   
                optimizer.step()  
                # optimizer.zero_grad()
                if n_count % 10 == 0:
                    print('Epoch: %d n_count: %5d loss: %.8f' % (epoch + 1, n_count + 1, loss.item()))

            avg_train_loss = epoch_loss / (n_count + 1)

            # ------------------ Validation phase ------------------
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch_yx in val_loader:
                    val_batch_x, val_batch_y = val_batch_yx[1].to(device), val_batch_yx[0].to(device)
                    val_out = model(val_batch_y)
                    val_loss += criterion(val_out, val_batch_x).item()

            val_loss /= len(val_loader)

             # ------------------Record loss value ------------------
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(val_loss)
            # ------------------------------------------------

            # ------------------ Logging ------------------
            output = '%4d %.5f %.5f' % (epoch + 1, epoch_loss / (n_count + 1), val_loss)
            f.write(output + '\n')

            elapsed_time = time.time() - start_time
            log('Epoch: %4d, Train Loss: %4.8f, Val Loss: %4.8f, Time: %4.4f s' %
                (epoch + 1, epoch_loss / (n_count + 1), val_loss, elapsed_time))

            # ------------------ Save the best model ------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, os.path.join(save_dir, 'best_model.pth'))
                log(f'Best model saved at epoch {epoch + 1} with Val Loss: {val_loss:.6f}')

            # Save each epoch's model
            torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))

        total_training_time = time.time() - total_start_time
        hours,remainder=divmod(total_training_time,3600)
        minutes,seconds=divmod(remainder,60)

        log(f'total Training Time:{int(hours)}h {int(minutes)}m {seconds:.2f}s')
            
    f.close()

# =================== Plot training curve ===================
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, 'k-', label='Training Loss', linewidth=4)
    plt.plot(val_loss_history, 'r-', label='Validation Loss', linewidth=2,linestyle='--')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path)
    log(f'Training curves saved to: {plot_path}')
    plt.show()
    plt.close()
    # =======================================================


