_base_ = '../mask_rcnn/mask-rcnn_r50-caffe_fpn_1x_coco.py'
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
            mask_head=dict(num_classes=2),),
    # model training and testing settings
    train_cfg=dict(rcnn=dict(mask_thr_binary=0.5),type='EpochBasedTrainLoop',max_epoch=50))

# 修改数据集相关配置
data_root = 'data/WenCounty/'
metainfo = {
    'classes': ('tree','massoniana'),
    'palette': [
        (220, 20, 60),
    ]
}
train_dataloader = dict(
    batch_size=1,
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

max_epochs = 50
train_cfg = dict(max_epochs=max_epochs)


# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/instance_val.json')
test_evaluator = val_evaluator

# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth'
