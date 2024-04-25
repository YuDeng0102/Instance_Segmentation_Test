_base_ = '../ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py'
# learning policy
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
# 设置类别
model = dict(
    type='MaskScoringRCNN',
    roi_head=dict(
        type='MaskScoringRoIHead',
        mask_iou_head=dict(
            type='MaskIoUHead',
            num_convs=4,
            num_fcs=2,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=2),
            bbox_head=dict(num_classes=2), 
            mask_head=dict(num_classes=2),))

# 修改数据集相关配置
data_root = 'data/GuiZhou/'
metainfo = {
    'classes': ('cedar','massoniana'),
    'palette': [
        (220, 20, 60),
    ]
}
train_dataloader = dict(
    batch_size=5,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')))
test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator



# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco/ms_rcnn_r50_caffe_fpn_2x_coco_bbox_mAP-0.388__segm_mAP-0.363_20200506_004738-ee87b137.pth'
