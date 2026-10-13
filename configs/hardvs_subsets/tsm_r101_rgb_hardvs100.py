# Same as tsm_r101_rgb_hardvs50.py on the HARDVS 100-class subset (seed 0).
_base_ = ['./tsm_r101_rgb_hardvs50.py']

ann_dir = 'data/hardvs/annotations/subsets/classes_100_seed0'
num_classes = 100

model = dict(cls_head=dict(num_classes=num_classes))
train_dataloader = dict(
    dataset=dict(ann_file=f'{ann_dir}/train_label_aps.csv'))
val_dataloader = dict(dataset=dict(ann_file=f'{ann_dir}/val_label_aps.csv'))
test_dataloader = dict(dataset=dict(ann_file=f'{ann_dir}/test_label_aps.csv'))
