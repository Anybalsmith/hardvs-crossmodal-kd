# Same as tsm_r101_event_hardvs50.py on the full HARDVS dataset (300 classes).
_base_ = ['./tsm_r101_event_hardvs50.py']

ann_dir = 'data/hardvs/annotations'
num_classes = 300

model = dict(cls_head=dict(num_classes=num_classes))
train_dataloader = dict(
    dataset=dict(ann_file=f'{ann_dir}/train_label_dvs.csv'))
val_dataloader = dict(dataset=dict(ann_file=f'{ann_dir}/val_label_dvs.csv'))
test_dataloader = dict(dataset=dict(ann_file=f'{ann_dir}/test_label_dvs.csv'))
