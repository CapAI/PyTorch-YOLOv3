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
-d MarineNet-resized \
-o test.txt \
-o train.txt \
-o valid.txt \
-f split.dvc \
python train_val_test_yolo.py --src /home/jupyter/PyTorch_YOLOv3/MarineNet-resized/data --nval 5000 --seed 23102019


# PyTorch YOLOv3
dvc run -d train.py \
-d MarineNet-resized \
-o logs \
-f train.dvc \
python train.py \
--model_def config/yolov3-custom.cfg \
--pretrained_weights weights/darknet53.conv.74 \
--batch_size 16 \
--n_cpu 7 \
--checkpoint_interval 1 \
--data_config config/custom_dvc.data

/home/jupyter/PyTorch_YOLOv3/checkpoints/1571848490/yolov3_ckpt_99.pth

dvc run -d train.py \
-d MarineNet-resized \
-f train_99.dvc \
python train.py \
--model_def config/yolov3-custom.cfg \
--pretrained_weights /home/jupyter/PyTorch_YOLOv3/checkpoints/1571848490/yolov3_ckpt_99.pth \
--batch_size 16 \
--n_cpu 7 \
--checkpoint_interval 1 \
--data_config config/custom_dvc.data

dvc run -d train.py \
-d MarineNet-resized \
-f train_99_65.dvc \
python train.py \
--model_def config/yolov3-custom.cfg \
--pretrained_weights /home/jupyter/PyTorch_YOLOv3/checkpoints/1572266506/yolov3_ckpt_65.pth \
--batch_size 16 \
--n_cpu 7 \
--checkpoint_interval 1 \
--data_config config/custom_dvc.data

# if the training is succesful log it in MLflow
# dvc run -d ../mlflow_scripts/mlflow_logging.py \
# -d MarineNet-resized \
# -o mlflow_out \
# python ../mlflow_scripts/mlflow_logging.py \
# --experiment_name train_yolov3 \
# --remoreurl 'http://35.204.159.181/' \
# --model_name 'yolov3' \
# --run_name 'fully retrained 100 epoch' \
# --params "{'epochs': 100, 'gradient_accumulations': 2, 'model_def': 'yolov3-custom.cfg', 'data_config': 'custom_dvc.data', 'pretrained_weights': 'darknet53.conv.74', 'img_size': 416, 'freeze_pretrained': 0, 'iou_thres': 0.5, 'conf_thres': 0.5, 'nms_thres': 0.5}" \
# --metrics "{'val_f1':0.6829, 'val_mAP':0.8084, 'val_precision':0.5542, 'val_recall':0.8895}" \
# --model '/home/jupyter/PyTorch_YOLOv3/checkpoints/1571848490/yolov3_ckpt_99.pth' \
# --dvc_path 'train.dvc' \
# --artifact_dir '/home/jupyter/PyTorch_YOLOv3/logs/1571848490/'


# 
dvc run -d test.py \
-d MarineNet-resized \
-f test.dvc \
python test.py \
--set test \
--model_def config/yolov3-custom.cfg \
--data_config config/custom_dvc.data \
--weights_path checkpoints/1571848490/yolov3_ckpt_99.pth \
--class_path data/custom/classes.names \
--iou_thres 0.5 \
--conf_thres 0.5 \
--nms_thres 0.5 \
--batch_size 16 \
--n_cpu 7


# Tensorboard
tensorboard --logdir /home/jupyter/PyTorch_YOLOv3/logs --port 6006

# mlflow
mlflow ui