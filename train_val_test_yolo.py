from tqdm import tqdm
import os
import random

random.seed(359244)

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='/home/jupyter/PyTorch_YOLOv3/MarineNet/data', help='source directory, that contains folders train/ and test/')
    parser.add_argument('--nval', type=int, default=5000, help='number of images to sample for validation (from the train folder), for reference ntest = 3250')
    opt = parser.parse_args()
    
#     SRC = os.path.join('/home','jupyter','MarineNet-45k')
    src_img = os.path.join(opt.src, 'train')
    src_img_test = os.path.join(opt.src, 'test')

#     N_VAL = 3250
    img_list = os.listdir(src_img)
    val_list = random.sample(img_list, opt.nval)
    val_list = [os.path.join(src_img,x)+'\n' for x in val_list]
    train_list = [os.path.join(src_img,x)+'\n' for x in list(set(img_list)-set(val_list))]
    test_list = [os.path.join(src_img_test, x)+'\n' for x in os.listdir(src_img_test)]

    with open(os.path.join(opt.src, 'train.txt'), 'w') as train:
        train.writelines(train_list)

    with open(os.path.join(opt.src, 'valid.txt'), 'w') as valid:
        valid.writelines(val_list)

    with open(os.path.join(opt.src, 'test.txt'), 'w') as test:
        test.writelines(test_list)

if __name__ == "__main__":
    main()
    
    
