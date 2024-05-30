_base_='../_base_/samseg-maskrcnn.py'

fold_num=0
work_dir=f'./work_dirs/samseg-maskrcnn/BJFU/fold_{fold_num}'
dataset_type = 'CocoDataset'
data_root = f'data/datasets_BJFU/fold_{fold_num}/'
test_root='data/datasets_BJFU/'
batch_size = 1

metainfo = {
    'classes': ('2','4','5','7','tree','1','3','6')
}

resume = False

base_lr=1e-4
num_things_classes =8
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
max_epochs = 30
num_queries =100
sam_pretrain_ckpt_path = "checkpoints/sam_vit_b_01ec64.pth"


model = dict(
    type='SAMSegMaskRCNN',
    backbone=dict(
        init_cfg=dict(
            checkpoint=sam_pretrain_ckpt_path, type='Pretrained'),
        type='Adapted_ImageEncoderViT'),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True
    )
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam',
        lr=base_lr,
        weight_decay=0.001
    )
)

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/')))
test_dataloader =  dict(
        dataset=dict(
        data_root=test_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/')))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = dict(ann_file=test_root + 'annotations/instances_test.json')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)