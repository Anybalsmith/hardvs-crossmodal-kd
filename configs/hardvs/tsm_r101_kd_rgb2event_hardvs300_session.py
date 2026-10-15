# Same as tsm_r101_kd_rgb2event_hardvs300.py on the session-based re-split of
# the full dataset (tools/data/hardvs/make_hardvs_splits.py, hour sessions,
# seed 0), with the RGB teacher trained by tsm_r101_rgb_hardvs300_session.py.
_base_ = ['./tsm_r101_kd_rgb2event_hardvs300.py']

ann_dir = 'data/hardvs/annotations/splits/session_hour_seed0'

model = dict(
    teacher_config='configs/hardvs/tsm_r101_rgb_hardvs300_session.py',
    teacher_checkpoint=
    'work_dirs/tsm_r101_rgb_hardvs300_session/best_acc_top1_epoch_XX.pth')
train_dataloader = dict(
    dataset=dict(
        ann_file=f'{ann_dir}/train_label_aps.csv',
        ann_file_event=f'{ann_dir}/train_label_dvs.csv'))
val_dataloader = dict(
    dataset=dict(
        ann_file=f'{ann_dir}/val_label_aps.csv',
        ann_file_event=f'{ann_dir}/val_label_dvs.csv'))
test_dataloader = dict(
    dataset=dict(
        ann_file=f'{ann_dir}/test_label_aps.csv',
        ann_file_event=f'{ann_dir}/test_label_dvs.csv'))
