from pycocotools.coco import COCO
import numpy as np
from pycocotools import mask
import pickle
import os

PATH_TO_DATASET = '../../../../Datasets/COCO'
NAME_ANNOTATIONS = 'instances_val2017.json'

def open_maps():

    coco = COCO(os.path.join(PATH_TO_DATASET, NAME_ANNOTATIONS))

    for img_idx in coco.imgs.keys():
        img_metadata = coco.loadImgs([img_idx])[0]
        print(img_metadata["file_name"])

        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_idx))


        all_masks = np.zeros((img_metadata['height'], img_metadata['width']))


        for i, instance in enumerate(cocotarget):
            rle = mask.frPyObjects(instance['segmentation'], img_metadata['height'], img_metadata['width'])
            m = mask.decode(rle)

            if len(m.shape) == 3:
                m = m.sum(axis=2)


            all_masks += m

        all_masks = np.where(all_masks != 0, 1,0)

        if not os.path.exists(os.path.join(PATH_TO_DATASET, 'maps')):
            os.mkdir(os.path.join(PATH_TO_DATASET, 'maps'))
        with open(os.path.join(PATH_TO_DATASET,  f'maps/{img_metadata["file_name"]}.pickle', 'wb')) as f:
            pickle.dump(all_masks,f)

def test_pickle(name_img):
    with open(os.path.join(PATH_TO_DATASET,f'maps/{name_img}.pickle'), 'rb') as f:
        map = pickle.load(f)
    print(map.shape, map.max())

if __name__ == '__main__':
    open_maps()

