#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse



def get_img_label_list(file_path, outfile):
    if os.path.isdir(file_path):
        files = os.listdir(file_path)
        write_file = open(outfile, 'w')
        for i, name in enumerate(files):
            if name.startswith('.'):
                continue
            img_path = file_path + '/' + name
            if os.path.isdir(img_path):
                imgs = os.listdir(img_path)
                for j, img in enumerate(imgs):
                    write_file.write(name + '/' + img + ' ' + str(i) + '\n')
        write_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generating image to label list for  dataset')
    parser.add_argument('-i', '--image-path', type=str, required=True, help='origin image file path')
    parser.add_argument('-o', '--out-path', type=str, required=True, help='output image file path', default='image_label.txt')
    args = parser.parse_args()

    get_img_label_list(args.image_path, args.out_path)

