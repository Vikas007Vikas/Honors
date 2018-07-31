import os
import numpy as np
from skimage import io
def load_image(path):
    img = io.imread(path)
#    print img.shape
    return img

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png','.jpg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False

def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory,f) for f in files if is_an_image_file(f)]

def load_images(path):
    train_paths = os.path.join(path,'train/01')
    depth_paths = os.path.join(path,'depth_maps/01')
    all_paths_train,all_paths_depth = list_image_files(train_paths),list_image_files(depth_paths)
    images_train,images_depth = [],[]
    images_train_paths,images_depth_paths = [],[]
    for each_train,each_depth in zip(all_paths_train,all_paths_depth):
        img_train,img_depth = load_image(each_train),load_image(each_depth)
        images_train.append(img_train)
        images_depth.append(img_depth)
        images_train_paths.append(each_train)
        images_depth_paths.append(each_depth)

    return {'images':np.array(images_train),
            'images_paths':np.array(images_train_paths),
            'depth_maps':np.array(images_depth),
            'depth_map_paths':np.array(images_depth_paths)
    }

data = load_images('./images')
