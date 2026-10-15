# RGB-only TSM-R101 on the HARDVS 50-class subset (seed 0).
# Also used as the teacher of tsm_r101_kd_rgb2event_hardvs50.py.
_base_ = [
    '../_base_/models/tsm_r101.py', '../_base_/schedule_sgd_40e.py',
    '../_base_/default_runtime.py'
]

# Paths are relative to the repository root (see README).
data_root = 'data/hardvs/rawframes'
ann_dir = 'data/hardvs/annotations/subsets/classes_50_seed0'
num_classes = 50

model = dict(cls_head=dict(num_classes=num_classes))

train_pipeline = [
    # Same event-density aware sampling as the event model, computed on the
    # paired DVS recording and applied to the RGB frames.
    dict(
        type='PairedEventDensitySampleFrames',
        num_clips=8,
        ratio=(6, 2),
        tau=0.95,
        low_sampling_mode='uniform'),
    dict(type='HARDVSFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=4,
        num_clips=8,
        test_mode=True),
    dict(type='HARDVSFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackActionInputs')
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='HARDVSFrameDataset',
        ann_file=f'{ann_dir}/train_label_aps.csv',
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='HARDVSFrameDataset',
        ann_file=f'{ann_dir}/val_label_aps.csv',
        data_prefix=dict(img=data_root),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='HARDVSFrameDataset',
        ann_file=f'{ann_dir}/test_label_aps.csv',
        data_prefix=dict(img=data_root),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator
