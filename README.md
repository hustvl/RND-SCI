# This is the demo code of our paper "A Range-Null Space Decomposition Approach for Fast and Flexible Spectral Compressive Imaging" in submission to ICCV 2023.

This repo includes:  

- Specification of dependencies.
- Training code.
- Evaluation code.
- Geting params and FLOPs code.
- Testing training time and inference speed code.
- Pre-trained models for RND_SAUNet.
- README file.

This repo can reproduce the main results in Tabel 1. and Tabel 2. of our main paper.
All the source code and pre-trained models will be released to the public for further research.


## 1. Create Environment:

------
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- [PyTorch >= 1.3](https://pytorch.org/)

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

  ```shell
  pip install -r requirements.txt
  ```

- Install cuda operation
  ```shell
  python setup.py develop
  ```

## 2. Prepare Dataset:

Download the dataset from https://github.com/mengziyi64/TSA-Net, put the dataset into the corresponding folder 'code/datasets/', and recollect them in the following form:

    |--datasets
        |--cave_1024_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
        |--CAVE_512_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
        |--KAIST_CVPR2021  
            |--1.mat
            |--2.mat
            ： 
            |--30.mat
        |--TSA_simu_data  
            |--mask.mat   
            |--Truth
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
        |--TSA_real_data  
            |--mask.mat   
            |--Measurements
                |--scene1.mat
                |--scene2.mat
                ： 
                |--scene5.mat
### 3. Training and Testing for simulation experiment:
#### Training 

##### Training from scratch
If you want to train any other model with `RND-SCI`, please refer to the following format:
    
    # RND_[model]
    python simu_train.py --method rnd_[original model name] --outf ./exp/simu_rnd_[original model name] / --seed 42 --gpu_id 0 

for example:

    # RND-SAUNet
    python simu_train.py --method rnd_saunet_1stg --outf ./exp/simu_rnd_saunet_1stg/ --seed 42 --gpu_id 0 

    # RND_MST
    python simu_train.py --method rnd_mst --outf ./exp/simu_rnd_mst/ --seed 42 --gpu_id 0 
    ...

##### Training with pre-trained model
If you have original pre-trained model without/ with RND-SCI framework and want to train with RND-SCI, please refer to the following format:

    python simu_train.py --method rnd_[original model name] --outf ./exp/simu_rnd_[original model name] / --pretrained_model_path  [your model with/without RND-SCI path] --seed 42 --gpu_id 0 

Please use checkpointing (--cp) when running out of memory. refer to 'utils/simu_utils/simu_args.py' to use more options.

#### Testing 
a). Test our models on the HSI dataset. The results will be saved in 'code/test/' in the MatFile format. For example, we test the RND-SAUNet:

    python simu_test.py --method rnd_saunet_1stg --outf ./test/simu_rnd_saunet_1stg   --pretrained_model_path [your model with/without RND-SCI path]

b). Calculate quality assessment. We use the same quality assessment code as DGSMP. So please use Matlab, get in 'code/analysis_tools/Quality_Metrics/', and then run 'Cal_quality_assessment.m'.

c). If you want test the other models with RND-SCI , please change the model your want to test in above step a).

d). If you want to plug and play a model with RND, whether the pre-trained model uses RND-SCI or not, you can **directly use the command** that refers to a).

#### 4. Training and Testing for real data experiment:
##### Training from scratch
If you want to train any other model with `RND-SCI`, please refer to the following format:
    
    # RND_[model]
    python real_train.py --method rnd_[original model name] --outf ./exp/real_rnd_[original model name] / --seed 42 --gpu_id 0 --isTrain

for example:

    # RND-SAUNet
    python real_train.py --method rnd_saunet_1stg --outf ./exp/real_rnd_saunet_1stg/ --seed 42 --gpu_id 0 --isTrain

    # RND_MST
    python real_train.py --method rnd_mst --outf ./exp/real_rnd_mst/ --seed 42 --gpu_id 0 --isTrain
    ...

##### Testing 
a). Test our models on the HSI dataset. The results will be saved in 'code/evaluation/testing_result/' in the MatFile format. For example, we test the RND-SAUNet:

    python real_test.py --method rnd_saunet_1stg --outf ./test/real_rnd_saunet_1stg  --pretrained_model_path [your model with/without RND-SCI path]

b). Calculate quality assessment. We use no reference image quality assessments (Naturalness Image Quality Evaluator, **NIQE** ). So please use Matlab, get in 'code/analysis_tools/Quality_Metrics/', and then run 'NIQE_metric.m'.

c). If you want test SAUNet-1stg or the others , please change the model your want to test in above step a).

d). If you want to plug and play a model with RND, whether the pre-trained model uses RND-SCI or not, you can **directly use the command** that refers to a).

#### 5. Get training time and inference FPS
##### Inference FPS
If we want to get inference fps of RND-SAUNet, run the following commond:

    python test_fps.py --method rnd_saunet_1stg --outf ./test/real_rnd_saunet_1stg --gpu_id 0
**Please mask sure that the GPU is not occupied by another program before running the commond.** Other models are similar to this.

##### Training time
Afer you finish the training of model, please run these commands:

    cd analysis_tools/
    python tranining_time [your training log path]

#### 6. Evaluating the Params and FLOPS of models
You can get the Params and FLOPS of models **at the begin of training**. Or use following commonds 
(for instance, we get these values of RND_SAUNet. Other methods are similar):

    python test_fps.py --method rnd_saunet_1stg

#### 7. This repo is mainly based on *the toolbox for Spectral Compressive Imaging*, which is provided by MST and contains 11 learning-based algorithms for spectral compressive imaging. 
The above toolbox offer us a fair benchmark comparison. We use the methods correspoding to original repo as follows:

(1)  TSA-Net: https://github.com/mengziyi64/TSA-Net

(2)  DGSMP: https://github.com/TaoHuang95/DGSMP

(3) GAP-Net: https://github.com/mengziyi64/GAP-net

(4) ADMM-Net: https://github.com/mengziyi64/ADMM-net

(5) HDNet: https://github.com/Huxiaowan/HDNet

(6) MST: https://github.com/caiyuanhao1998/MST

(7) CST: https://github.com/caiyuanhao1998/MST

(8) DAUHST: https://github.com/caiyuanhao1998/MST

(9) SAUNet: https://github.com/hustvl/SAUNet

We thank these repos and have cited these works in our manuscript.