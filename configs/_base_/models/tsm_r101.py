model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained='torchvision://resnet101',
        depth=101,
        norm_eval=False,
        shift_div=8,
        num_segments=8,
        is_shift=True),
    cls_head=dict(
        type='TSMHead',
        num_classes=50,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True,
        num_segments=8,
        average_clips='prob'),
    # Normalisation is done per modality in the pipelines
    # (NormalizePerModality), hence no mean/std here.
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=None,
        std=None,
        to_rgb=False,
        to_float32=True,
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None)
