import os
import numpy as np
import shutil




def processing_dataset(path_to_dataset, path_to_annotations, path_output):
    os.mkdir(path_output)
    os.mkdir(os.path.join(path_output, 'img_dir'))
    os.mkdir(os.path.join(path_output, 'ann_dir'))

    path_to_file = os.path.join(path_to_dataset, 'ImageSets', 'Main', 'trainval.txt')
    path_to_imgs = os.path.join(path_to_dataset, 'JPEGImages')
    path_to_segmaps = path_to_annotations

    img_outs = os.path.join(path_output, 'img_dir')
    ann_outs = os.path.join(path_output, 'ann_dir')

    with open(path_to_file, 'r') as f:
        for name_img in f.readlines():
            name_img = name_img[:-1]
            path_to_img = os.path.join(path_to_imgs, f'{name_img}.jpg')
            img_out = os.path.join(img_outs, f'{name_img}.jpg')
            path_to_segmap = os.path.join(path_to_segmaps, f'{name_img}.png')
            segmaps_out = os.path.join(ann_outs, f'{name_img}.png')
            shutil.copy2(path_to_img, img_out)
            shutil.copy2(path_to_segmap, segmaps_out)
    shutil.copy2(os.path.join(path_to_dataset, 'ImageSets', 'Main', 'train.txt'),  os.path.join(path_output, 'train.txt'))
    shutil.copy2(os.path.join(path_to_dataset, 'ImageSets', 'Main', 'val.txt'),  os.path.join(path_output, 'val.txt'))

if __name__ == '__main__':
    path_to_dataset = '/Users/adels/Downloads/VOCdevkit/VOC2010'
    path_to_annotations = '/Users/adels/Downloads/pascal-context/33_context_labels'
    path_output = '../../data/pascal_context'
    processing_dataset(path_to_dataset, path_to_annotations,path_output)




