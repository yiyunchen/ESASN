# ESASN

This Code is a PyTorch implementation for Our Paper "Lightweight Single Image Super-Resolution through Efficient Second-order Attention Spindle Network"

## Prerequisites:
- Python 3.6
- PyTorch 0.4
- numpy
- skimage
- imageio
- matplotlib
- tqdm

Our code is based on [RCAN](https://github.com/yulunzhang/RCAN).
About how to calculate Mult-Adds, you can check the [torchsummaryX](https://github.com/nmhkahn/torchsummaryX).

## Train

Cd to 'ESASN_TrainCode/code',run the following scripts to train models

```
CUDA_VISIBLE_DEVICES=0 python main.py --scale 2 --save ESASN_R6G6F48W96B16g4_BIx2lr4 --model ESASN --epochs 1000 --batch_size 16 --patch_size 96 --n_resgroups 6 --n_resblocks 6 --n_feats 48 --w_feats 96 --lr 4e-4 --ext sep --reset --n_val 10 --groups 4
```

## Test
Cd to 'ESASN_TestCode/code', run the following scripts.

```
CUDA_VISIBLE_DEVICES=2 python main.py --data_test MyImage --scale 2 --model ESASN --n_resgroups 6 --n_resblocks 6 --n_feats 48 --w_feats 96 --groups 4 --pre_train ../model/ESASN_R6G6F48W96x2_best.pt --test_only --save_results --save 'ESASNB' --testpath ../LR/LRBI --testset Urban100
```

Run 'Evaluate_PSNR_SSIM.m' to obtain PSNR/SSIM values.


## BibTeX

```
@INPROCEEDINGS{9102946,

  author={Y. {Chen} and Y. {Chen} and J. -H. {Xue} and W. {Yang} and Q. {Liao}},

  booktitle={2020 IEEE International Conference on Multimedia and Expo (ICME)}, 

  title={Lightweight Single Image Super-Resolution Through Efficient Second-Order Attention Spindle Network}, 

  year={2020},

  volume={},

  number={},

  pages={1-6},

  doi={10.1109/ICME46284.2020.9102946}}
```




