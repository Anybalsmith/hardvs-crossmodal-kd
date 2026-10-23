# Common settings of the RGB (APS) TSM-R50 models on the HRI subsets of HARDVS
# (human-robot / drone command gestures), same recipe as the event models:
#   - 8 segments, one frame each, frame_interval=1 so that the whole recording
#     is covered (frame_interval=4 only ever samples the first frame(s) of the
#     2-4 PNG recordings),
#   - MultiScaleCrop + resize 224, NO horizontal flip (the subsets hold
#     left/right pairs, a flip would turn a class into its mirror),
#   - test: one clip of 8 frames, centre crop 224,
#   - best checkpoint on val top-1; test every 5 epochs (TestDuringTrainHook).
# Children set `split` (annotation folder) and `num_classes`.
_base_ = [
    '../_base_/models/tsm_r101.py', '../_base_/schedule_sgd_40e.py',
    '../_base_/default_runtime.py'
]

# Paths are relative to the repository root (see README, HRI subsets).
data_root = 'data/hardvs/rawframes'
split = 'hri_v4_official'
ann_dir = 'data/hardvs/annotations/splits/' + split
num_classes = 5

model = dict(
    backbone=dict(pretrained='torchvision://resnet50', depth=50),
    cls_head=dict(num_classes=num_classes))

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
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
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='HARDVSFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackActionInputs')
]

# batch 16 (two R50 runs must fit on one A6000 beside other jobs); the lr is
# auto-scaled from 0.02 @ 128.
train_dataloader = dict(
    batch_size=16,
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
        pipeline=test_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=32,
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

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=1, save_best='acc/top1'))
