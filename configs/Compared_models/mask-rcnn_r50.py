_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]



# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[22, 34],
#         gamma=0.1)
# ]
# 设置类别
num_classes=1
max_epochs=40
# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes), mask_head=dict(num_classes=num_classes)))



# 修改数据集相关配置
metainfo = {
    'classes': ('crown',)
}


work_dir=f'./work_dirs/mask-rcnn/data--/'
dataset_type = 'CocoDataset'
data_root = 'data/data--/'
batch_size = 4



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
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/')))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/instances_test.json')


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=4)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')



# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
#load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
