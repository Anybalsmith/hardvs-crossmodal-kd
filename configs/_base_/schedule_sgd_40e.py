train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=40, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=35,
        eta_min=1e-5,
        by_epoch=True,
        begin=5,
        end=40,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=2e-5),
    clip_grad=dict(max_norm=20, norm_type=2))

# lr is given for a total batch size of 128 and scaled linearly to the actual
# one: batch 32 on 1 GPU -> lr 0.005.
auto_scale_lr = dict(enable=True, base_batch_size=128)
