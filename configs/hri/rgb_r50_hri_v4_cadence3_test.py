# Diagnostic, test only: evaluate a v4 checkpoint with EVERY test recording
# reduced to 3 frames repeated over the 8 segments. If the model learnt
# "repeated frames -> A129", nearly all test clips go to A129, including the
# long A129 recordings it classifies elsewhere on the normal test.
#   python tools/test.py configs/hri/rgb_r50_hri_v4_cadence3_test.py <ckpt> --dump <out.pkl>
_base_ = ['./_base_hri_rgb_r50.py']

test_pipeline = [
    dict(
        type='CadenceAugSampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True,
        num_distinct=(3, ),
        force_test=True),
    dict(type='HARDVSFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackActionInputs')
]
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
