#!/bin/bash
python work.py --cfg_path=./configs/Compared_models/Cascade-mask-rcnn_MyCoco7.py --work_dir=./work_dirs/Cascade-mask-rcnn_MyCoco7 --data_root=./data/MyCoco7
python work.py --cfg_path=./configs/Compared_models/mask-rcnn_r50_MyCoco7.py --work_dir=./work_dirs/mask-rcnn_MyCoco7 --data_root=./data/MyCoco7
python work.py --cfg_path=./configs/Compared_models/mask-scoring-rcnn_MyCoco7.py --work_dir=./work_dirs/mask-scoring-rcnn_MyCoco7 --data_root=./data/MyCoco7
python work.py --cfg_path=./configs/Compared_models/solov2_MyCoco7.py --work_dir=./work_dirs/solov2_MyCoco7 --data_root=./data/MyCoco7
python work.py --cfg_path=./configs/Compared_models/Yolact_MyCoco7.py --work_dir=./work_dirs/Yolact_MyCoco7 --data_root=./data/MyCoco7