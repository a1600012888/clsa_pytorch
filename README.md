## Unoffical implementation of Contrastive Learning with Stronger Augmentations. WIP

This is an unofficial PyTorch implementation of the [CLSA paper](https://openreview.net/forum?id=KJSC_AsN14):

Note: This implementation is most adopted from the offical MoCo's implementation from https://github.com/facebookresearch/moco 
This repo aims to be minimal modifications on that code. 



### Preparation
Note: This section is copied from MoCo's repo

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo aims to be minimal modifications on that code. Check the modifications by:
```
diff main_moco.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
diff main_lincls.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
```


### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_clsa.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --mlp --aug-plus --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```
This script uses all the default hyper-parameters as described in CLSA paper.


### Linear Classification
Note: This section is copied from MoCo's repo

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

### TODO:
1. ImageNet-1K CLSA-Single-200epoch pretraining: Running
2. ImageNet-1K CLSA-Mul-200epoch pretraining:  Running
3. Evaluate CLSA-Single/-Mul on ImageNet Linear Protocal 
4. Evaluate CLSA-Single/-Mul on VOC07 Det
