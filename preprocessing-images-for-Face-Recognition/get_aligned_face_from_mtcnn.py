from src import detect_faces, show_bboxes
from PIL import Image
import cv2
import os
import pdb
import argparse
import numpy as np
from src.align_trans import get_reference_facial_points, warp_and_crop_face
from tqdm import tqdm


def get_aligned_images(origin_path, output_path, crop_size, replace=False):
    if os.path.isdir(origin_path):
        files = os.listdir(origin_path)
        for idx, file in enumerate(files):

            # get images in each file
            imgs_root = os.path.join(origin_path, file)
            # print(imgs_root)
            if os.path.isdir(imgs_root):
                for img_path in os.listdir(imgs_root):

                    img = Image.open(os.path.join(imgs_root, img_path))

                    # preprocessing image

                    bounding_boxes, landmarks = detect_faces(img)
                    img_cv2 = np.array(img)[..., ::-1]
                    if len(bounding_boxes) and len(landmarks):
                        for i in range(len(bounding_boxes)):
                            box = bounding_boxes[i][:4].astype(np.int32).tolist()
                            for idx, coord in enumerate(box[:2]):
                                if coord > 1:
                                    box[idx] -= 1
                            if box[2] + 1 < img_cv2.shape[1]:
                                box[2] += 1
                            if box[3] + 1 < img_cv2.shape[0]:
                                box[3] += 1
                            face = img_cv2[box[1]:box[3], box[0]:box[2]]
                            if face.shape[0] <= 0 or face.shape[1] <= 0:
                                continue
                            landmark = landmarks[i]
                            facial5points = [[landmark[j] - box[0], landmark[j + 5] - box[1]] for j in range(5)]
                            dst_img = warp_and_crop_face(face, facial5points, crop_size=crop_size)
                            dst_img = Image.fromarray(dst_img[..., ::-1])

                            save_root = os.path.join(output_path, 'normal_aligned', str(file))
                            save_path = os.path.join(save_root, img_path)
                            if not os.path.exists(save_root):
                                os.makedirs(save_root)
                            print('normal_save_path = ', save_path)
                            dst_img.save(save_path)
                    else:

                        print('there no bounding box and landmarks detected for image {}_{}'.format(imgs_root, img_path))
                        save_root = os.path.join(output_path, 'unnormal_aligned', str(file))
                        save_path = os.path.join(save_root, img_path)
                        # pdb.set_trace()
                        print('except_path = ', save_path)    # 'images/test_output_file/none_bounding_boxes_imgs/Darryl_Stingley_0001.jpg'
                        # pdb.set_trace()
                        if not os.path.exists(save_root):
                            os.makedirs(save_root)
                        print('unnormal_save_path = ', save_path)
                        img.save(save_path)

                        continue



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='data preprocessing for face recognition')
    # parser.add_argument('-ops', '--output-size', nargs="+", type=int, default=112, help='output image size for MTCNN')
    # parser.add_argument('-ipr', '--input-root', type=str, default='images/origin_images', help='origin images fold path')
    # parser.add_argument('-opr', '--output-root', type=str, default='images/test_output_file',
    #                     help='output images fold path')
    # args = parser.parse_args()
    # print('args = ', args)
    # output_size = tuple(args.output_size)

    get_aligned_images('/mnt/lustre/share/platform/dataset/LFW/lfw', 'images/test_output_file', crop_size=(112, 112))
    # get_aligned_images(origin_path=args.input_root, output_path=args.output_root, crop_size=output_size)
