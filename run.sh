#!/bin/bash
# python work.py --cfg_path=./configs/Compared_models/Cascade-mask-rcnn_MyCoco7.py --work_dir=./work_dirs/Cascade-mask-rcnn_MyCoco7 --data_root=./data/MyCoco7
# python work.py --cfg_path=./configs/Compared_models/mask-rcnn_r50_MyCoco7.py --work_dir=./work_dirs/mask-rcnn_MyCoco7 --data_root=./data/MyCoco7
# python work.py --cfg_path=./configs/Compared_models/mask-scoring-rcnn_MyCoco7.py --work_dir=./work_dirs/mask-scoring-rcnn_MyCoco7 --data_root=./data/MyCoco7
# python work.py --cfg_path=./configs/Compared_models/solov2_MyCoco7.py --work_dir=./work_dirs/solov2_MyCoco7 --data_root=./data/MyCoco7
# python work.py --cfg_path=./configs/Compared_models/Yolact_MyCoco7.py --work_dir=./work_dirs/Yolact_MyCoco7 --data_root=./data/MyCoco7
python work.py --cfg_path=configs/Adapted_SAM/samseg-mask2former_BJFU.py --data_root=./data/datasets_BJFU --work_dir=samseg-mask2former_BJFU --batch_size=3

python work.py --cfg_path=configs/Adapted_SAM/Adapted-samseg-mask2former_BJFU.py --data_root=./data/datasets_BJFU --work_dir=Adapted-samseg-mask2former_BJFU --batch_size=1
python work.py --cfg_path=configs/Adapted_SAM/Adapted-samseg-maskrcnn_BJFU.py --data_root=./data/datasets_BJFU --work_dir=Adapted-samseg-maskrcnn_BJFU --batch_size=1
python work.py --cfg_path=./configs/Adapted_SAM/Adapted-samseg-mask2former_MyCoco7.py --data_root=./data/MyCoco7 --work_dir=./work_dirs/Adapted-samseg-mask2former_MyCoco7
python work.py --cfg_path=configs/Adapted_SAM/samseg-mask2former_MyCoco7.py --data_root=./data/MyCoco7 --work_dir=samseg-mask2former_MyCoco7 --batch_size=3
python work.py --cfg_path=configs/Adapted_SAM/samseg-maskrcnn_MyCoco7.py --data_root=./data/MyCoco7 --work_dir=samseg-maskrcnn_MyCoco7 --batch_size=6

shutdown




