# rgb_r50_hri_v4.py plus background removal: everything outside the
# clip-level person mask (YOLO11x-seg, union over the recording, dilated
# 15 px, edge softened) is set to black, right after decoding, in the train,
# val and test pipelines. Recordings where YOLO finds nobody are unchanged.
# Masks: tools/data/hardvs/make_clip_masks.py.
_base_ = ['./_base_hri_rgb_r50.py']

data_root = 'data/hardvs/rawframes'
frame_op = dict(
    type='PersonMaskBackground',
    mask_root='data/hardvs/person_masks/clip_masks_yolo_d15',
    data_root=data_root,
    feather=7)

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='HARDVSFrameDecode'),
    frame_op,
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
    frame_op,
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
