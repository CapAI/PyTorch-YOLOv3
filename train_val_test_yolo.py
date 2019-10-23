from tqdm import tqdm
import os
import random
import argparse


def main(src='/home/jupyter/PyTorch_YOLOv3/MarineNet/data', nval=5000, output_dir=''):
    src_img = os.path.join(src, 'train')
    src_img_test = os.path.join(src, 'test')

    img_list = os.listdir(src_img)
    val_list = random.sample(img_list, nval)
    val_list = [os.path.join(src_img,x)+'\n' for x in val_list]
    train_list = [os.path.join(src_img,x)+'\n' for x in list(set(img_list)-set(val_list))]
    test_list = [os.path.join(src_img_test, x)+'\n' for x in os.listdir(src_img_test)]

    with open(os.path.join(output_dir, 'train.txt'), 'w') as train:
        train.writelines(train_list)

    with open(os.path.join(output_dir, 'valid.txt'), 'w') as valid:
        valid.writelines(val_list)

    with open(os.path.join(output_dir, 'test.txt'), 'w') as test:
        test.writelines(test_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='/home/jupyter/PyTorch_YOLOv3/MarineNet/data', help='source directory, that contains folders train/ and test/')
    parser.add_argument('--output_dir', type=str, default='', help='destination to output the text files')
    parser.add_argument('--nval', type=int, default=5000, help='number of images to sample for validation (from the train folder), for reference ntest = 3250')
    parser.add_argument('--seed', type=int, default=23102019)
    opt = parser.parse_args()
    
    random.seed(opt.seed)
    
    main(src=opt.src, nval=opt.nval, output_dir=opt.output_dir)
