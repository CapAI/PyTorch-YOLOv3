import os,sys
import cv2
from tqdm import tqdm
import argparse

def resize(src='/home/jupyter/PyTorch_YOLOv3/MarineNet/data', size=700):

    # Set the directory you want to start from
    phases = ['train','test']
#     img_folder = 'image_2'
#     dest_folder = 'image_yolo'
    for phase in phases:
        src_dir = os.path.join(src, phase)
        dst_dir = src_dir.replace('MarineNet', 'MarineNet-resized')
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for fn in tqdm(os.listdir(src_dir)):
            if fn[-4:].lower() not in ['.jpg','.png','.jpeg']:
                continue
            f_path = os.path.join(src_dir, fn)
            dest_path = os.path.join(dst_dir, fn)
            img = cv2.imread(f_path)
            h, w, _ = img.shape
            if h > size or w > size:
                scale_h = size/h
                scale_w = size/w
                scale = min(scale_h,scale_w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(dest_path, img)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='/home/jupyter/PyTorch_YOLOv3/MarineNet/data', help='source directory, that contains folders "train/" and "test/".')
    parser.add_argument('--size', type=int, default=700, help='maximum width or height of image.')
    opt = parser.parse_args()
    
    resize(src=opt.src, size=opt.size)