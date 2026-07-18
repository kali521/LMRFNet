# LMRFNet 

LMRFNet is a network model based on Python 3.9, designed to remove the complex noise of DAS-VSP.

## 1. Simple installation from pypi.org

Install the Python libraries required for training the LMRFNet and testing the DAS-VSP denoising process

`pip install -`

  `pip install python3.9`

  `pip install pytorch1.31.1`

  `pip install matplotlib`

  `pip install einops`

  `pip install numpy`

The above command should directly install all the dependencies required for the fully functional version of LMRFNet. You don't need to manually download anything.


## 2. simple example

To illustrate how to denoise DAS-VSP data using LMRFNet, let's start with a simple example.


### 2.1 Testing(denoise_test copy.py)

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


## Dataset Construction Workflow

## Data Availability 

Confidentiality restrictions prevent public release of the raw dataset. To improve reproducibility, this section describes the construction and partitioning procedures for the signal set, noise set, training set, and validation set.
## 3. Signal Set

Construct velocity models using Tesseral software and perform forward modeling on these models to generate synthetic DAS‑VSP signals. Build 20 velocity models and obtain synthetic DAS‑VSP signal data through forward modeling.
### 3.1 Velocity Model Construction

Construct multiple types of velocity models according to actual geological conditions, including horizontal, thin‑layer, fault, tilted, anticline, and flexure models. Adjust parameters such as offset for each type to produce multiple velocity models, totaling 20 models. Take the horizontal velocity model as an example: set length to 1 km and depth to 2 km, divided into 7 uniform geological layers. Increase the P‑wave velocity from 1000 m/s to 4000 m/s layer by layer. Use the software default settings for formation media, with density automatically adjusted according to velocity parameters.
### 3.2 Observation System Setup

Set the wellhead horizontal position to 800 m. Deploy 2000 receivers within the depth range of 1–2000 m underground, with a receiver spacing of 1 m. Place the source on the surface. Use a zero‑phase Ricker wavelet with a dominant frequency range of 40–80 Hz. Position the source on the left side of the wellhead, with a horizontal offset of 100–300 m from the wellhead.
### 3.3 Forward Modeling
After completing the velocity models, observation system, and forward modeling parameter settings, apply the elastic wave equation to perform forward modeling for each velocity model individually. Set the sampling interval to 0.4 ms and the recording duration to 2 s. Obtain the corresponding clean DAS‑VSP signal data from forward modeling, with synthetic seismic records of size 5000 × 2000. Normalize the clean synthetic data by the maximum value. Apply a sliding window of size 80 × 80 with a step size of 20 to partition the records into blocks, yielding 40,737 clean signal blocks of size 80 × 80.
## 4. Noise Set

Extract noise segments from real field seismic data to construct the noise set.
### 4.1 Noise Set Construction

Select publicly available field seismic data with a sampling frequency of 2500 Hz and a receiver spacing of 1 m. From these field data, selectively intercept horizontal noise, random noise, low‑frequency noise, and fading noise. Normalize each extracted noise segment by its maximum value. Apply a sliding window of size 80 × 80 with a step size of 20 to extract noise blocks, obtaining 40,737 noise blocks of size 80 × 80. Assign equal probability to the four noise types.
## 5. Noisy Dataset

Synthesize the noisy data by combining signal data and noise data.
### 5.1 Partitioning of Noisy Dataset

Split the entire dataset into training set and test set at a ratio of 8:2. Divide the signal dataset according to the velocity models: use forward‑modeled data from 16 velocity models as the training set and data from the remaining 4 velocity models as the validation set. Split the noise dataset randomly according to noise types into training and validation sets at the same 8:2 ratio, ensuring equal probability distribution of the four noise types in both sets. Ensure no overlap between the training set and the validation set.
### 5.2 Construction of Noisy Dataset

To produce noisy data with a distribution of different SNRs, multiply each noise block by a random factor uniformly selected from (0, 4]. Then add each resulting noise block to each corresponding clean signal block, generating 40,737 noisy data blocks with varying SNR levels. Ensure a one‑to‑one correspondence between the noisy dataset and the clean signal dataset.


