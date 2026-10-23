# HRI v4 restricted to the recordings favourable to RGB (good exposure of the
# person, more than 10 PNGs i.e. APS at 75-125 fps instead of 10 fps), split by
# recording day: annotations data/hardvs/annotations/splits/hri_v4_rgbfav
# (370 / 65 / 204; made by make_favorable_subset.py in the tsm_UniFormer HRI
# work, copy the folder from there). Same settings as rgb_r50_hri_v4.py.
# Train and test share no recording day; val is a random 15 % of the training
# days and only monitors training: report the test at the LAST epoch
# (<work_dir>/<timestamp>/test_metrics.jsonl), not at the best val epoch.
_base_ = ['./_base_hri_rgb_r50.py']

ann_dir = 'data/hardvs/annotations/splits/hri_v4_rgbfav'
train_dataloader = dict(dataset=dict(ann_file=f'{ann_dir}/train_label_aps.csv'))
val_dataloader = dict(dataset=dict(ann_file=f'{ann_dir}/val_label_aps.csv'))
test_dataloader = dict(dataset=dict(ann_file=f'{ann_dir}/test_label_aps.csv'))
