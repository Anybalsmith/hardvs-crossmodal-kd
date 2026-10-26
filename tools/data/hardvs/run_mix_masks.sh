#!/bin/bash
# K processes of make_mix_masks.py sharing one GPU, then the merge. Run from the repository root:
#   tools/data/hardvs/run_mix_masks.sh data/hardvs/annotations/splits/hri_v1_official 2 [extra args]
# Logs: <out>/shard<i>.log; the merge prints 'done' at the end of <out>/merge.log.
ANN=$1; K=${2:-2}; shift 2
OUT=data/hardvs/person_masks/mix_sam2_$(basename $ANN)
mkdir -p $OUT
for i in $(seq 0 $((K - 1))); do
  python tools/data/hardvs/make_mix_masks.py --ann-dir $ANN --out $OUT --shard $i --num-shards $K "$@" > $OUT/shard$i.log 2>&1 &
done
wait
python tools/data/hardvs/make_mix_masks.py --ann-dir $ANN --out $OUT --merge > $OUT/merge.log 2>&1
tail -2 $OUT/merge.log
