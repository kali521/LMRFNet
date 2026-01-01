# LMRFNet 

LMRFNet is a network model based on Python 3.9, designed to remove the complex noise of DAS-VSP.

## Simple installation from pypi.org

Install the Python libraries required for training the LMRFNet and testing the DAS-VSP denoising process

`pip install -`

  `pip install python3.9`

  `pip install pytorch1.31.1`

  `pip install matplotlib`

  `pip install einops`

  `pip install numpy`

The above command should directly install all the dependencies required for the fully functional version of LMRFNet. You don't need to manually download anything.


## A simple example

To illustrate how to denoise DAS-VSP data using LMRFNet, let's start with a simple example.


### 1. Testing(denoise_test copy.py)

First, import the DAS-VSP data.

`seismic_noisy = loadmat('/home/zhangzeyuan1/d2sm-master/quzaoshiyan/Data1.mat')`
`seismic_noisy = seismic_noisy['data']  #test`

Load `from Cut_combine import cut`to trim the DAS-VSP seismic data into test blocks of size patch-size by patch-size.

`patch-size=80`

Next, call the trained LMRFNet denoising model. The model path is:

`parser.add_argument('--model_name', default='model_150.pth', type=str, help='the model name')`
`model = torch.load(os.path.join(args.model_dir, args.model_name), weights_only=False)`

Then, load the denoised test block with `from Cut_combine import combine` to restore it to the entire DAS-VSP seismic data.

`seismic_predict = combine(predict_datas, patch_size, strides_x, strides_y, seismic_block_h, seismic_block_w)`

The noise removal of the entire DAS-VSP seismic data is saved at:

`savemat('/home/zhangzeyuan1/d2sm-master/quzaoshiyan/predict.mat', seismic_dict)`


## Dataset

The noise in this experiment comes from real DAS-VSP seismic recordings. Therefore, the dataset is confidential and cannot be made public.


