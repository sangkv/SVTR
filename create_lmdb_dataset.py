import os
import glob
from pathlib import Path

import lmdb
import cv2
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)


def get_list_path(path):
    exts = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    list_path = []
    for ext in exts:
        list_ext = glob.glob(os.path.join(path, '**', ext), recursive=True)
        list_path.extend(list_ext)
    return list_path


def read_data_from_folder(folder_path):
    list_path = get_list_path(folder_path)

    image_path_list = []
    label_list = []

    for path_image in list_path:
        try:
            path_txt = Path(path_image).with_suffix('.txt')
            with open(path_txt, 'r', encoding='utf-8') as f:
                text = f.read().rstrip('\n')
                image_path_list.append(path_image)
                label_list.append(text)
        except:
            print('Error in file path: ', path_image)
            continue
    
    return image_path_list, label_list


def show_data(image_path_list, label_list, demo_number=5):
    for i in range(demo_number):
        print ('image: %s\nlabel: %s\n' % (image_path_list[i], label_list[i]))


if __name__ == '__main__':
    input_path = 'data/train'
    output_path = 'data/trainset'

    image_path_list, label_list = read_data_from_folder(input_path)
    show_data(image_path_list, label_list)
    createDataset(output_path, image_path_list, label_list)

    input_path = 'data/val'
    output_path = 'data/valset'

    image_path_list, label_list = read_data_from_folder(input_path)
    show_data(image_path_list, label_list)
    createDataset(output_path, image_path_list, label_list)
    
    print("DONE!")

