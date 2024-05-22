_base_ = '../yolact/yolact_r50_1xb8-55e_coco.py'


# 设置epoch和学习率
max_epochs = 100
num_classes=7
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
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
        num_classes=num_classes),
    mask_head=dict(
        type='YOLACTProtonet',
        num_classes=num_classes,        
        )
)

# 修改数据集相关配置



load_from='https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r101_1x8_coco/yolact_r101_1x8_coco_20200908-4cbe9101.pth'