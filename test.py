from __future__ import division

import os
import sys
import time
import datetime
import argparse
from tqdm import tqdm

sys.path.append('..')

import mlflow
import git
import yaml

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

from mlflow_scripts.mlflow_logging import log_mlflow

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, coco=False):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
#         print('targets: \r\n', targets[:,1])
        if coco:
            targets[:,1] = 8 # target boat in coco yolo labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
#             print('outputs: \r\n', outputs)
#             print('targets: \r\n', targets)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class #, true_positives, pred_scores, pred_labels

# def logMLflow():
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--set", type=str, default='valid', help="use {train, valid, test} set")
    opt = parser.parse_args()
    print(opt)
    
    parameters = vars(opt)
    metrics = {}
#     mlflow.log_params()
    
#     repo = git.Repo(search_parent_directories=True)
#     sha = repo.head.object.hexsha
#     mlflow.log_param('git_hexsha', sha)
    
#     with open('MarineNet.dvc', 'r') as f:
#         dvc = yaml.safe_load(f)
#     mlflow.log_param('dvc_md5', dvc['md5'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config[opt.set]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")
    
    coco = opt.class_path[-len('coco.names'):]=='coco.names'

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
        coco=coco
    )
    mAP = AP.mean()
    
    metrics['mAP'] = mAP
    
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        metrics[f'AP_{class_names[c]}'] = AP[i]
        metrics[f'precision_{class_names[c]}'] = precision[i]
        metrics[f'recall_{class_names[c]}'] = recall[i]
        metrics[f'f1_{class_names[c]}'] = f1[i] 

    print(f"mAP: {mAP}")
    
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    
    weight_name = opt.weights_path.split('/')[-1]
    def_name = opt.model_def.split('/')[-1].split('.')[0]
    
    log_mlflow(experiment_name = f'eval_{opt.set}',
              remoteurl='http://35.204.159.181/',
              model_name = def_name,
              run_name = weight_name,
              params = parameters,
              metrics = metrics,
              model = opt.weights_path,
              dvc_path = 'train.dvc', #TODO check which dvc file to use
              artifact_dir=None) # add examples of detections 
    
#     print(f'TP : {true_positives}')
#     print(f'pred_scores: {pred_scores}')
#     print(f'pred_labels: {pred_labels}')