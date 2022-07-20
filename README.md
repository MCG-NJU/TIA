<div align="center">
  <h1>Task-specific Inconsistency Alignment for Domain Adaptive Object Detection <br> (CVPR 2022)</h1>
</div>

<div align="center">
  <h3><a>Liang Zhao</a>, <a href=https://wanglimin.github.io/>Limin Wang</a></h3>
</div>

<div align="center">
  <h4> <a href=https://arxiv.org/abs/2203.15345>[paper]</a></h4>
</div>

## :heavy_check_mark: Requirements
* Ubuntu 16.04
* Python 3.7
* [CUDA 10.1](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.2.0](https://pytorch.org)
* [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)

Remember to compile the cuda dependencies using following simple commands following [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0):
```bash
cd lib
python setup.py build develop
```

## :books: Datasets
```bash
cd data
bash download_datasets.sh
```
Still under construction, we will upload to google drive soon!

You can also prepare all the datasets following [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn), [SCL](https://github.com/harsh-99/SCL) and [HTCN](https://github.com/chaoqichen/HTCN).
It is also important to note that we have written all the codes for **Pascal VOC** format.
Meanwhile, the pixel-level adaptation/interpolation/augmentation is implemented via [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) following [HTCN](https://github.com/chaoqichen/HTCN), for a fair comparison.
We use the **default** parameters for training, and perhaps better parameters or networks would give better results. 

## :bell: Pre-trained Models

Follow the convention, two pre-trained models on ImageNet, i.e., VGG16 and ResNet101 are employed. 
Please download and place these two models in `pretrained_models/` from:
* **VGG16:** [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

## :deciduous_tree: Authors' checkpoints

```bash
cd checkpoints
bash download_checkpoints_tia.sh
```
The file structure should be as follows:

    TIA/
    ├── cfgs/
    ├── data/
    ├── lib/
    ├── pretrained_models/
    │   ├── resnet101_caffe.pth
    │   └── vgg16_caffe.pth
    ├── models/
    │   ├── cityscale/vgg16/TIA/da_faster_rcnn_cityscape_7.pth
    │   ├── clipart/res101/TIA/da_faster_rcnn_clipart_7.pth
    │   └── kitti2cityscape/vgg16/TIA/da_faster_rcnn_kitti2cityscape_7.pth
    ├── scripts/
    _init_paths.py
    parse.py
    README.md
    test_net.py
    trainval_net.py

## :pushpin: Quick start: testing scripts
Run the following command directly to test:
```bash
bash scripts/{dataset_name}/test.sh
```
For example, to test TIA on the clipart dataset:
```bash
bash scripts/clipart/test.sh
```

## :fire: Training scripts
Run the following command directly to train:
```bash
bash scripts/{dataset_name}/train.sh
```
For example, to train TIA on the clipart dataset:
```bash
bash scripts/clipart/train.sh
```

## :question: Q & A
Q: Why are the training results unstable and inconsistent?

A: It is firstly, I think, a intrinsic problem of DAOD. Since GRL-based adversarial training is essentially a compromise of GAN on large models like detectors. Therefore, DAOD and the primary GAN possess similar properties, both of their training are unstable, occasionally even mode collapse, and neither of their losses are indicative, making it difficult for us to determine their convergence.
Second, the pixel-level adaptation, or namely the data augmentation, further reinforces this instability.
As we simply wrap both the original and generated target-like images into the source domain, which could confuse the discriminator's judgment,
even though this shows the best performance. 
Finally, the multi-view learning like TIA relies to some extent on the randomness of initialization.
If there are large discrepancies between different classifiers or localizers, then they will capture even higher inconsistencies.
We have tried to manually constrain the initialization with some success, and if you are interested in this, you can also try more diverse initialization constraints.

Q: Why are there some inconsistencies between the test results of the provided models and the results reported in the paper?

A: After the paper was submitted, I attempted to impose some constraints against the initialized randomness in the adaptation from cityscape to foggy cityscape and obtained better results. Also, I tried to overcome the severe overfitting problem in the adaptation from kitti to cityscape and found more stable results with the help of earlystopping.

Q: Why is the open source so late?

A: I suffer from severe procrastination. If no one pushes me, I will keep procrastinating :), and if someone pushes me, I will keep procrastinating with pressure :(.

## :mag: Related repos
Our project references the codes in the following repos:

* Chen _et al_., [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn), [DA-Faster Rcnn-PyTorch](https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch)
* Shen _et al_., [SCL](https://github.com/harsh-99/SCL)
* Xu _et al_., [CRDA](https://github.com/megvii-research/CR-DA-DET)
* Chen _et al_., [HTCN](https://github.com/chaoqichen/HTCN)

## :scroll: Citing TIA
If you find our code or paper useful to your research work, please consider citing our work using the following bibtex:
```bibtex
@inproceedings{tiadaod22cvpr,
  author    = {Liang Zhao and
               Limin Wang},
  title     = {Task-specific Inconsistency Alignment for Domain Adaptive Object Detection},
  booktitle = {{CVPR}},
  year      = {2022}
}
```
```bibtex
@article{zhao2022tia,
  author    = {Zhao, Liang and 
               Wang, Limin},
  title     = {Task-specific Inconsistency Alignment for Domain Adaptive Object Detection},
  journal   = {arXiv preprint arXiv:2203.15345},
  year      = {2022}
}
```
