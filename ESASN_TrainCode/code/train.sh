CUDA_VISIBLE_DEVICES=1 python main.py --scale 2 --save ESASN_R6G6F48W96B16g4_BIx2lr4 --model ESASN --epochs 1000 --batch_size 16 --patch_size 96 --n_resgroups 6 --n_resblocks 6 --n_feats 48 --w_feats 96 --lr 4e-4 --ext sep --reset --n_val 10 --groups 4

