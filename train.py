from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from tensorboardX import SummaryWriter



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=7, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=True, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("-freeze_pretrained", action='store_true', help="keep first 75 layers fixed, (re)train layers 75-106")
    opt = parser.parse_args()
    print(opt)
    
    parameters = vars(opt)
    metrics = {}

    t = round(time.time())
    parameters['id'] = t
    
    writer = SummaryWriter(f'Xlogs/{t}')
    logger = Logger(f"logs/{t}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs(f"checkpoints/{t}", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    MEANS = [0.48581678, 0.50988996, 0.52099264]
    STDS = [0.25317615, 0.24736303, 0.26655522]
    if MEANS is None or STDS is None:
        get_stats = MeanStd(train_path=train_path)
        MEANS, STDS = get_stats.get_means_stds()
        print(f'means: {MEANS}')
        print(f'stds: {STDS}')
    
    
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)
    
    print(model)
    
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training, means=MEANS, stds=STDS)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    freeze_limit = 75
    parameters['freeze_limit'] = freeze_limit
    if opt.freeze_pretrained:
        parameters = list()
        print('length of mod_list: ', len(model.module_list), type(model.module_list))
        for m, mod in enumerate(model.module_list):
            if m < freeze_limit:
                for param in mod.parameters():
                    param.requires_grad_(False)
            else:
                parameters.append(mod.parameters())
    else:
        parameters = model.parameters()

        #     parameters = list(model.module_list[106].parameters()) + list(model.module_list[105].parameters()) + list(model.module_list[94].parameters()) + list(model.module_list[93].parameters()) + list(model.module_list[82].parameters()) + list(model.module_list[81].parameters())
    
    optimizer = torch.optim.Adam(parameters)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    breaking = False
    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        
        if breaking:
            break
            
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]
                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            writer.add_scalar('Total loss',loss.item(), batches_done)

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)
            
            

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            writer.add_scalars('eval_metrics', 
                            {"val_precision": precision.mean(),
                            "val_recall": recall.mean(),
                            "val_mAP": AP.mean(),
                            "val_f1": f1.mean()}, epoch)
            
            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/{t}/yolov3_ckpt_%d.pth" % epoch)

    
    metrics['val_precision'] = precision.mean()
    metrics['val_recall'] = recall.mean()
    metrics['val_mAP'] = AP.mean()
    metrics['val_f1'] = f1.mean()
    
    model_name = opt.model_def.split('/')[-1].split('.')[0]
    weights_name = opt.pretrained_weights.split('/')[-1]
    weights_name = weight_name + f'_frozen_{freeze_limit}' if opt.freeze_pretrained else weight_name
    weights_path = os.path.join('checkpoints', t, f'yolov3_ckpt_{epoch}.pth' )
    
    log_mlflow(experiment_name = f'train_obj_detect',
              remoteurl='http://35.204.159.181/',
              model_name = model_name,
              run_name = weights_name,
              params = parameters,
              metrics = metrics,
              model = weights_path,
              dvc_path = 'train.dvc', #TODO check which dvc file to use
              artifact_dir=f"logs/{t}")