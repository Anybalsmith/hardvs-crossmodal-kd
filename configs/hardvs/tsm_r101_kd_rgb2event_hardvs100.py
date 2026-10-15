# Same as tsm_r101_kd_rgb2event_hardvs50.py on the HARDVS 100-class subset
# (seed 0), with the RGB teacher trained by tsm_r101_rgb_hardvs100.py.
_base_ = ['./tsm_r101_kd_rgb2event_hardvs50.py']

ann_dir = 'data/hardvs/annotations/subsets/classes_100_seed0'
num_classes = 100

model = dict(
    teacher_config='configs/hardvs/tsm_r101_rgb_hardvs100.py',
    teacher_checkpoint=
    'work_dirs/tsm_r101_rgb_hardvs100/best_acc_top1_epoch_XX.pth',
    cls_head=dict(num_classes=num_classes))
train_dataloader = dict(
    dataset=dict(
        ann_file=f'{ann_dir}/train_label_aps.csv',
        ann_file_event=f'{ann_dir}/train_label_dvs.csv',
        num_classes=num_classes))
val_dataloader = dict(
    dataset=dict(
        ann_file=f'{ann_dir}/val_label_aps.csv',
        ann_file_event=f'{ann_dir}/val_label_dvs.csv',
        num_classes=num_classes))
test_dataloader = dict(
    dataset=dict(
        ann_file=f'{ann_dir}/test_label_aps.csv',
        ann_file_event=f'{ann_dir}/test_label_dvs.csv',
        num_classes=num_classes))
