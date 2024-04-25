_base_ = '../yolact/yolact_r50_1xb8-55e_coco.py'


# 设置epoch和学习率
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
        milestones=[16, 22],
        gamma=0.1)
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    bbox_head=dict(
        type='YOLACTHead',
        num_classes=1),
    mask_head=dict(
        type='YOLACTProtonet',
        num_classes=1,        
        )
)

# 修改数据集相关配置
data_root = 'data/WenCounty/'
metainfo = {
    'classes': ('tree',),
    'palette': [
        (220, 20, 60),
    ]
}
train_dataloader = dict(
    batch_size=5,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instance_train.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instance_val.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/instance_val.json')
test_evaluator = val_evaluator


load_from='https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r101_1x8_coco/yolact_r101_1x8_coco_20200908-4cbe9101.pth'