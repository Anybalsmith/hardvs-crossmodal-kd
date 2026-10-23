# rgb_r50_hri_v4.py with cadence augmentation in training: half of the long
# recordings are reduced to 3 or 4 frames, as the 2-4 PNG recordings, so that
# repeated frames stop being a class cue (in the v4 training set every
# recording with <= 3 PNGs is A129). Validation and test are unchanged.
_base_ = ['./_base_hri_rgb_r50.py']

train_pipeline = [
    dict(
        type='CadenceAugSampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        prob=0.5,
        num_distinct=(3, 4)),
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
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
