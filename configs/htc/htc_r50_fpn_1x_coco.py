_base_ = '../cascade_rcnn/htc-without-semantic_r50_fpn_1x_coco.py'

max_epochs = 60
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[32, 48],
        gamma=0.1)
]



model = dict(
    data_preprocessor=dict(pad_seg=True),
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            seg_scale_factor=1 / 8,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=183,
            loss_seg=dict(
                type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2))))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img='train2017/', seg='stuffthingmaps/train2017/'),
        pipeline=train_pipeline))
