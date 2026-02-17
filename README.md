# Deep Softtriple hashing for Multi-Label cross-modal retrieval [Paper]( https://www.sciencedirect.com/science/article/pii/S0893608026001759)
This paper is accepted for Neural Networks(NN).

## Training

### Processing dataset
Before training, you need to download the oringal data from [coco](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)(include 2017 train,val and annotations), nuswide [Google drive](https://drive.google.com/file/d/11w3J98uL_KHWn9j22GeKWc5K_AYM5U3V/view?usp=drive_link), mirflickr25k [Baidu, code: u9e1](https://pan.baidu.com/s/1upgnBNNVfBzMiIET9zPfZQ) or [Google drive](https://drive.google.com/file/d/18oGgziSwhRzKlAjbqNZfj-HuYzbxWYTh/view?usp=sharing) (include mirflickr25k and mirflickr25k_annotations_v080), then use the "dataset/make_XXX.py" to generate .mat file. The generated data is available from [Baidu, code: wyb8](https://pan.baidu.com/s/17QeFeZgTAOiY9qe0wo8OxQ).

After all mat file generated, the dir of `dataset` will like this:
~~~
dataset
├── base.py
├── __init__.py
├── dataloader.py
├── coco
│   ├── caption.mat 
│   ├── index.mat
│   └── label.mat 
├── flickr25k
│   ├── caption.mat
│   ├── index.mat
│   └── label.mat
└── nuswide
    ├── caption.txt  # Notice! It is a txt file!
    ├── index.mat 
    └── label.mat
~~~

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start

After the dataset has been prepared, we could run the follow command to train.
> python main.py

## Citation 
If you find this useful for your research, please use the following.

```
@article{HAN2026108713,
title = {Deep Softtriple hashing for Multi-Label cross-modal retrieval},
author = {Shuo Han and Qibing Qin and Jinkui Hou and Wenfeng Zhang and Lei Huang},
journal = {Neural Networks},
volume = {198},
pages = {108713},
year = {2026},
issn = {0893-6080},
url = {https://www.sciencedirect.com/science/article/pii/S0893608026001759},
doi = {https://doi.org/10.1016/j.neunet.2026.108713},
}
```

## Acknowledegements
[DCHMT](https://github.com/kalenforn/DCHMT)


