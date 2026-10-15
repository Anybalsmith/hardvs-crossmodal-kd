# RGB -> Event cross-modal distillation on the HARDVS 50-class subset (seed
# 0). Student: event TSM-R101 (same setup as tsm_r101_event_hardvs50.py).
# Teacher: frozen RGB TSM-R101 trained with tsm_r101_rgb_hardvs50.py.
_base_ = [
    '../_base_/models/tsm_r101.py', '../_base_/schedule_sgd_40e.py',
    '../_base_/default_runtime.py'
]

# Paths are relative to the repository root (see README).
data_root = 'data/hardvs/rawframes'
ann_dir = 'data/hardvs/annotations/subsets/classes_50_seed0'
num_classes = 50

# Only the `model` of the teacher config is used.
teacher_config = 'configs/hardvs/tsm_r101_rgb_hardvs50.py'
# Checkpoint of the trained RGB baseline, usually given on the command line:
#   --cfg-options model.teacher_checkpoint=<rgb_work_dir>/best_acc_top1_epoch_XX.pth
teacher_checkpoint = 'work_dirs/tsm_r101_rgb_hardvs50/best_acc_top1_epoch_XX.pth'

model = dict(
    type='KDCrossModalRecognizer2D',
    teacher_config=teacher_config,
    teacher_checkpoint=teacher_checkpoint,
    cls_loss_weight=1.0,
    # Logit KD: KL(teacher || student) at temperature 4.
    kd_loss_type='kl',
    kd_temperature=4.0,
    kd_loss_weight=0.3,
    # Feature KD: attention transfer on the last backbone feature map.
    feature_kd_loss_type='attention',
    feature_kd_loss_weight=0.5,
    # Both KD terms are weighted per sample by the teacher confidence.
    kd_sample_weight_mode='teacher_confidence',
    teacher_confidence_temperature=1.0,
    apply_kd_weight_to_feature=True,
    cls_head=dict(num_classes=num_classes))

# Student (event) pipelines: identical to the event-only baseline.
student_train_pipeline = [
    dict(
        type='EventDensityAwareSampleFrames',
        num_clips=8,
        ratio=(6, 2),
        tau=0.95,
        low_sampling_mode='uniform'),
    dict(type='HARDVSFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, ),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackActionInputs')
]
# Teacher (RGB) pipeline. The frame indices drawn by SampleFrames are
# replaced by the student's ones (remapped to the RGB timeline), and the
# student's crop/flip are replayed (sync_spatial_aug=True).
teacher_train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=4, num_clips=8),
    dict(type='HARDVSFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, ),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='NormalizePerModality'),
    dict(type='PackActionInputs')
]
# Validation / test: same deterministic sampling and center crop for both.
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

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='crossmodal_collate'),
    dataset=dict(
        type='HARDVSPairedDataset',
        ann_file=f'{ann_dir}/train_label_aps.csv',
        ann_file_event=f'{ann_dir}/train_label_dvs.csv',
        data_prefix=dict(img=data_root),
        num_classes=num_classes,
        pipeline_student=student_train_pipeline,
        pipeline_teacher=teacher_train_pipeline,
        sync_spatial_aug=True))
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='crossmodal_collate'),
    dataset=dict(
        type='HARDVSPairedDataset',
        ann_file=f'{ann_dir}/val_label_aps.csv',
        ann_file_event=f'{ann_dir}/val_label_dvs.csv',
        data_prefix=dict(img=data_root),
        num_classes=num_classes,
        pipeline_student=val_pipeline,
        pipeline_teacher=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='crossmodal_collate'),
    dataset=dict(
        type='HARDVSPairedDataset',
        ann_file=f'{ann_dir}/test_label_aps.csv',
        ann_file_event=f'{ann_dir}/test_label_dvs.csv',
        data_prefix=dict(img=data_root),
        num_classes=num_classes,
        pipeline_student=val_pipeline,
        pipeline_teacher=val_pipeline,
        test_mode=True))

# The model predicts both student and teacher scores: AccMetric reports
# acc/student_top1 and acc/teacher_top1.
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(save_best='acc/student_top1'))
