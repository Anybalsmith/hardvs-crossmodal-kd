default_scope = 'mmaction'
# Registers the HARDVS datasets/transforms, the KD recognizer and the hook.
custom_imports = dict(imports=['hardvs_kd'], allow_failed_imports=False)

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=3, max_keep_ckpts=3,
        save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

# Evaluate on the test split every 5 epochs and after the last one
# (results also appended to <work_dir>/<timestamp>/test_metrics.jsonl).
custom_hooks = [dict(type='TestDuringTrainHook', interval=5, test_last=True)]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)
log_level = 'INFO'

load_from = None
resume = False
randomness = dict(seed=42, deterministic=True, diff_rank_seed=False)
