cd PyTorch_YOLOv3

# conda env
conda activate torch

# resize
dvc run -d MarineNet/data/ \
-o MarineNet/data/ \
-f resize.dvc \
python resize_images.py --src /media/jupyter/PyTorch_YOLOv3/MarineNet-45k/data --size 700

# check if bounding boxes are not too small

# split data
dvc run -d MarineNet/data \
-o MarineNet/data/test.txt \
-o MarineNet/data/train.txt \
-o MarineNet/data/valid.txt \
-f split.dvc \
python train_val_test_yolo.py


# PyTorch YOLOv3
python train.py \
--model_def config/yolov3-custom.cfg \
--pretrained_weights weights/darknet53.conv.74 \
--batch_size 16 \
--n_cpu 7 \
--checkpoint_interval 1 \
--data_config config/custom_dvc.data


python test.py \
--set test \
--model_def config/yolov3-custom.cfg \
--data_config config/custom.data \
--weights_path yolov3_ckpt_99.pth \
--class_path data/custom/classes.names \
--iou_thres 0.5 \
--conf_thres 0.001 \
--nms_thres 0.5 \
--batch_size 16 \
--n_cpu 1 


# Tensorboard
tensorboard --logdir /home/jupyter/PyTorch_YOLOv3/logs --port 6006