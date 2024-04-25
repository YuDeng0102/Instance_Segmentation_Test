from mmengine import Config
import os
import subprocess

def prepare_config(base_config_path, fold_index, data_root,work_dir):
    cfg = Config.fromfile(base_config_path)
    
    # 数据相关
    cfg.data_root=os.path.join(data_root,f'fold_{fold_index}')
    cfg.train_dataloader.dataset.data_root=cfg.data_root
    cfg.train_dataloader.dataset.ann_file='annotations/instances_train.json'
    cfg.val_dataloader.dataset.data_root=cfg.data_root
    cfg.val_dataloader.dataset.ann_file='annotations/instances_val.json'
    cfg.test_dataloader.dataset.data_root=data_root
    cfg.test_dataloader.dataset.ann_file='annotations/instances_val.json'
    cfg.val_evaluator.ann_file=os.path.join(cfg.data_root,'annotations','instances_val.json')
    cfg.test_evaluator.ann_file=os.path.join(data_root,'annotations','test_annotations.json')

    #训练设置相关
    # cfg.max_epoch=50
    # cfg.train_cfg.max_epochs=50
    cfg.train_cfg.val_interval=10
    cfg.default_hooks.checkpoint=dict( 
        interval=1,  # 验证间隔
        max_keep_ckpts=1,  # 最多保存多少个权重文件
        save_best='coco/segm_mAP',  # 按照该指标保存最优模型
        type='CheckpointHook')

  

    # 更新工作目录，每个折保存自己的输出
   
    cfg.work_dir = f'{work_dir}/fold_{fold_index}'

    return cfg
def main(args):
    work_dir=args.work_dir
    cfg_path=args.cfg_path
    data_root=args.data_root
    for i in range(5):
        cfg=prepare_config(cfg_path,0,data_root,work_dir)
        cfg.dump(cfg_path) 
        if args.train==True:
            subprocess.run(["python", "tools/train.py", cfg_path])
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--train', default=True, type=bool,
                        help='是否执行训练')
    parser.add_argument('--test', default=False, type=bool,
                        help='是否进行测试')
    
    parser.add_argument('--cfg_path',default='configs/MyCoco5/solov2.py',type=str,
                        help='config文件的路径')
    parser.add_argument('--work_dir',default='./work_dirs/solov2',type=str,
                        help='work_dir的路径')
    parser.add_argument('--data_root',default='data/MyCoco5',type=str,
                        help='work_dir的路径')
    
    main(parser)
