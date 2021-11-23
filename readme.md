# Cerberus Transformer: Joint Semantic, Affordance and Attribute Parsing


[**Paper**](https://arxiv.org/abs/2109.05566) 

![teaser](doc/teaser.PNG)

![sup](doc/sup1.PNG)

![sup](doc/sup2.PNG)

![sup](doc/sup3.PNG)

![sup](doc/sup4.PNG)

![sup](doc/sup5.PNG)

![sup](doc/sup6.png)


## Introduction

Multi-task indoor scene understanding is widely considered as an intriguing formulation, as the affinity of different tasks may lead to improved performance. In this paper, we tackle the new problem of joint semantic, affordance and attribute parsing. However, successfully resolving it requires a model to capture long-range dependency, learn from weakly aligned data and properly balance sub-tasks during training. To this end, we propose an attention-based architecture named Cerberus and a tailored training framework. Our method effectively addresses aforementioned challenges and achieves state-of-the-art performance on all three tasks. Moreover, an in-depth analysis shows concept affinity consistent with human cognition, which inspires us to explore the possibility of extremely low-shot learning. Surprisingly, Cerberus achieves strong results using only 0.1\%-1\% annotation. Visualizations further confirm that this success is credited to common attention maps across tasks. Code and models are publicly available.


## Citation

If you find our work useful in your research, please consider citing:

    @article{chen2021pq,
    title={PQ-Transformer: Jointly Parsing 3D Objects and Layouts from Point Clouds},
    author={Chen, Xiaoxue and Zhao, Hao and Zhou, Guyue and Zhang, Ya-Qin},
    journal={arXiv preprint arXiv:2109.05566},
    year={2021}
    }


## Installation

### Requirements
    
    python =3.6
    CUDA>=10.1
    Pytorch>=1.3
    matplotlib
    opencv-python
    tensorboard
    termcolor
    plyfile
    trimesh>=2.35.39
    networkx>=2.2
    scripy
    

### Data preparation

#### Attribute

#### Affordance

#### Semantic

For 3D detection on ScanNet, follow the [README](https://github.com/facebookresearch/votenet/blob/master/scannet/README.md) under the `scannet` folder.

For layout estimation on ScanNet, download the sceneCAD layout dataset from 
[HERE](http://kaldir.vc.in.tum.de/scannet_planes).  Unzip it into `/path/to/project/scannet/`.

## Run Pre-trained Model

You can download pre-trained model [HERE](https://drive.google.com/file/d/1yawlsprl-bhRotpZS29inQo4f4ZSZSY-/view?usp=sharing).
Move the file to the project root path (`/path/to/project/pretrained_model`) and then evaluate the model with:

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 eval.py --checkpoint_path /path/to/project/pretrained_model/ckpt_epoch_last.pth



## Training and evaluating

To train a Cerberus on NYUd2 with a single GPU:

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py  --log_dir log/[log_dir] --pc_loss
    
To test the trained model with its checkpoint:

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 eval.py  --log_dir [log_dir] --checkpoint_path [checkpoint_path]



