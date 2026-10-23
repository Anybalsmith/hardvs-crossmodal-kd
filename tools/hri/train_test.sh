#!/bin/bash
# Train a config, then test its best-val checkpoint and dump the predictions.
# Run from the repository root:  tools/hri/train_test.sh configs/hri/rgb_r50_hri_v4_cadence.py [GPU id]
# Outputs: work_dirs/<config name>/ (logs, best_acc_top1_epoch_*.pth),
#          work_dirs/<config name>/test_best_<epoch>/results.pkl
set -e
CFG=$1; GPU=${2:-0}
N=$(basename $CFG .py); W=work_dirs/$N
export CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p $W
python tools/train.py $CFG --work-dir $W > $W/train.out 2>&1
B=$(ls $W/best_acc_top1_epoch_*.pth); E=$(basename $B .pth | sed 's/.*_//')
mkdir -p $W/test_best_$E
python tools/test.py $CFG $B --work-dir $W/test_best_$E --dump $W/test_best_$E/results.pkl > $W/test_best_$E/test.out 2>&1
echo "$(date '+%d/%m %H:%M') DONE $N best epoch $E"
