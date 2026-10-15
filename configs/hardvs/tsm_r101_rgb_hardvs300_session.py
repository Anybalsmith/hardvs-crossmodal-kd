# Same as tsm_r101_rgb_hardvs300.py on the session-based re-split of the full
# dataset (tools/data/hardvs/make_hardvs_splits.py, hour sessions, seed 0).
_base_ = ['./tsm_r101_rgb_hardvs300.py']

ann_dir = 'data/hardvs/annotations/splits/session_hour_seed0'

train_dataloader = dict(
    dataset=dict(ann_file=f'{ann_dir}/train_label_aps.csv'))
val_dataloader = dict(dataset=dict(ann_file=f'{ann_dir}/val_label_aps.csv'))
test_dataloader = dict(dataset=dict(ann_file=f'{ann_dir}/test_label_aps.csv'))
