
_base_='../_base_/samseg-maskrcnn.py'

work_dir=f'./work_dirs/samseg-maskrcnn/BJFU'
dataset_type = 'CocoDataset'
data_root = f'data/datasets_BJFU/'
test_root='data/datasets_BJFU/'
batch_size = 1

metainfo = {
    'classes': ('Tree',)
}

resume = False

base_lr=1e-2
num_things_classes =1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
max_epochs = 40

sam_pretrain_name = "sam-vit-large"
sam_pretrain_ckpt_path = "checkpoints/sam_vit_l_0b3195.pth"

model = dict(
    type='SAMSegMaskRCNN',
    backbone=dict(
        type='Adapted_ImageEncoderViT',
        embed_dim=1024,
        depth=24,
        num_heads=16,
        global_attn_indexes=[5, 11, 17, 23],
        init_cfg=dict(
            checkpoint=sam_pretrain_ckpt_path, type='Pretrained'),
    ),
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

optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.01, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=max_epochs,
        gamma=0.1,
        milestones=[
            23,
            35,
        ],
        type='MultiStepLR'),
]
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
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=3)
load_from=None