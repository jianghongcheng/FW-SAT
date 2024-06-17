# FAW-SAT 

### Flexible Window-based Self-attention Transformer in Thermal Image Super-Resolution [[Paper Link]](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/html/Jiang_Flexible_Window-based_Self-attention_Transformer_in_Thermal_Image_Super-Resolution_CVPRW_2024_paper.html)
[Hongcheng Jiang](https://github.com/jianghongcheng/), [Zhiqiang Chen](https://sse.umkc.edu/profiles/zhiqiang-chen.html)



## Citations
#### BibTeX

@InProceedings{Jiang_2024_CVPR,
    author    = {Jiang, Hongcheng and Chen, Zhiqiang},
    title     = {Flexible Window-based Self-attention Transformer in Thermal Image Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3076-3085}
}

## Environment
- [PyTorch >= 2.0](https://pytorch.org/) 
- [BasicSR == 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 

### Installation
Install Pytorh
```
pip install -r requirements.txt
```

## How To Test
- The test command is like

```
python predict.py 
```


## How To Train
- Refer to `dataloader.py` for dataset path to train.
- Refer to `option.yaml` for the configuration file of the model to train.
- The training command is like
```
python train.py 
```

## Trained Models

- Refer to `net_best_8.pth`for X8 SR
- Refer to `net_best_16.pth`for X16 SR

## Contact
If you have any question, please email hjq44@mail.umkc.edu 

