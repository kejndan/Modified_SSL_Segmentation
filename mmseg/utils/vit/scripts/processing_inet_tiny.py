import os
import shutil

path_to_dataset = '../../data/tiny-imagenet-200'

out_path = '../../data/imagenet_tiny'

os.mkdir(out_path)


def process_train():
    os.mkdir(os.path.join(out_path, 'train'))
    for cls in os.listdir(os.path.join(path_to_dataset, 'train')):
        if os.path.isdir(os.path.join(path_to_dataset, 'train', cls)):
            new_folder = os.path.join(out_path, 'train', cls)

            shutil.copytree(os.path.join(path_to_dataset, 'train', cls, 'images'), new_folder)


def process_test():
    os.mkdir(os.path.join(out_path, 'test'))
    labels = os.path.join(path_to_dataset, 'test', 'val_annotations.txt')
    with open(labels, 'r') as f:
        for row in f.readlines():
            name, label = row.split()[:2]

            new_path = os.path.join(out_path, 'test', label)

            if not os.path.exists(new_path):
                os.mkdir(new_path)

            shutil.copy2(os.path.join(path_to_dataset, 'test', 'images', name), os.path.join(new_path, name))

            print(name, label)


process_train()
process_test()
