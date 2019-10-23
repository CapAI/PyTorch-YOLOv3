cd PyTorch_YOLOv3

# conda env
conda activate torch

# resize
dvc run -d resize_images.py \
-d MarineNet \
-o MarineNet-resized \
-f resize.dvc \
python resize_images.py --src /home/jupyter/PyTorch_YOLOv3/MarineNet/data --size 700

# check if bounding boxes are not too small

# split data
dvc run -d train_val_test_yolo.py \
-d MarineNet-resize \
-o test.txt \
-o train.txt \
-o valid.txt \
-f split.dvc \
python train_val_test_yolo.py --src /home/jupyter/PyTorch_YOLOv3/MarineNet-resized/data --nval 5000 --seed 23102019


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
--conf_thres 0.5 \
--nms_thres 0.5 \
--batch_size 16 \
--n_cpu 1 


# Tensorboard
tensorboard --logdir /home/jupyter/PyTorch_YOLOv3/logs --port 6006