_base_='../_base_/samseg-mask2former.py'

fold_num=0
work_dir=f'./work_dirs/samseg-mask2former/BJFU/fold_{fold_num}'
dataset_type = 'CocoDataset'
data_root = f'data/datasets_BJFU/fold_{fold_num}/'
test_root='data/datasets_BJFU/'
batch_size = 1
fold_dir=f'fold_{fold_num}'
train_and_val_dataroot=data_root+fold_dir+'/'
metainfo = {
    'classes': ('1','2','3','4','5','6','7')
}



resume = False

base_lr=1e-4
num_things_classes =7
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
max_epochs = 30
num_queries =70

model = dict(
    type='SAMSegMask2Former',
    panoptic_head=dict(
        type='Mask2FormerHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=num_queries,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
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
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='test/')))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = dict(ann_file=test_root + 'annotations/instances_test.json')


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)