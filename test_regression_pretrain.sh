#!/bin/bash

python -m tfmplayground.external_priors \
--lib mouse \
--save_path /data/PFN/Mouse/prior_mouse.h5 \
--num_batches 20000 \
--batch_size 5 \
--prior_type mouse \
--min_features 3 \
--max_features 3 \
--min_seq_len 10 \
--max_seq_len 50 \
--max_classes 0

# python pretrain_regression.py \
# --epochs 100 \
# --steps 5000 \
# --batchsize 1 \
# --accumulate 25 \
# --priordump /data/PFN/Mouse/prior_mouse.h5 \
# --saveweights /data/PFN/Mouse/pretrained_mousepfn.pth
# --loadcheckpoint /data/PFN/Mouse/pretrained_tabpfn.pth

# python -m tfmplayground.external_priors \
# --lib tabpfn \
# --save_path /data/PFN/Mouse/prior_tabpfn.h5 \
# --num_batches 4000 \
# --batch_size 4 \
# --prior_type prior_bag \
# --min_features 3 \
# --max_features 3 \
# --min_seq_len 500 \
# --max_seq_len 2500 \
# --max_classes 0

# python pretrain_regression.py \
# --epochs 80 \
# --steps 400 \
# --batchsize 2 \
# --accumulate 2 \
# --priordump prior_tabpfn.h5 \
# --saveweights /data/PFN/Mouse/pretrained_tabpfn.pth