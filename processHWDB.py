# -*- coding: utf-8 -*-
import codecs

import struct
import numpy as np
import cv2
import os
import random

def transform_labelmap_to_utf8(src_path, src_encode, dst_path, dst_encode = 'utf-8'):
    '''
    :param src_path: source file path.
    :param src_encode:  source file encoding.
    :param dst_path:  destination file path.
    :param dst_encode: destination file encoding.
    :return:
    '''
    with codecs.open(src_path, mode='r', encoding=src_encode) as infile:
        lines = infile.readlines()
        with codecs.open(dst_path, mode='w', encoding=dst_encode) as outfile:
            outfile.writelines(lines)
    print('Transform encodings done!')

def decode_gnt_to_imgs(gnt):
    '''
    :param gnt: a writer's encoded ground truth file.
    :return:    samples list, each sample with format (charname, img)
    '''
    samples = []
    with codecs.open(gnt, mode='rb') as fin:
        while(True):
            left_cache = fin.read(4)
            if len(left_cache) < 4:
                break
            sample_size = struct.unpack("I", left_cache)[0]
            tag_code = str(fin.read(2), 'gbk')
            width = struct.unpack("H", fin.read(2))[0]
            height = struct.unpack("H", fin.read(2))[0]

            img = np.zeros(shape=[height, width], dtype=np.uint8)
            for r in range(height):
                for c in range(width):
                    img[r, c] = struct.unpack("B", fin.read(1))[0]
            if width*height + 10 != sample_size:
                break
            samples.append((tag_code[0], img))

    return samples

def init_save_dir(labelmap, save_root):
    '''
    initialize saveing dirs. save time in process.
    :param labelmap:
    :param save_root:
    :return:
    '''
    with open(labelmap, 'r') as infile:
        for line in infile.readlines():
            if line[0] == '.':
                os.mkdir(os.path.join(save_root, 'dot'))
            elif line[0] == '/':
                os.mkdir(os.path.join(save_root, 'slash'))
            else:
                os.mkdir(os.path.join(save_root, line[0]))


def gen_train_test_sets(img_root_dir, labelmap, train_index, test_index, subsetnum = None):
    samples = []
    labels = []
    subdirs = os.listdir(img_root_dir)
    subdirs.sort()

    if subsetnum is None:
        subsetnum = len(subdirs)

    for id, folder in enumerate(subdirs[:subsetnum]):
        for img in os.listdir(os.path.join(img_root_dir, folder)):
            samples.append(folder+'/'+ img + ' ' + str(id))
        # labels.append(folder+','+str(id))
        labels.append(folder)

    random.shuffle(samples)
    split_pos = int(0.8*len(samples))
    with open(train_index, 'w') as trainset:
        for sample in samples[: split_pos]:
            trainset.writelines(sample+'\n')

    with open(test_index, 'w') as testset:
        for sample in samples[split_pos:]:
            testset.writelines(sample+'\n')

    with open(labelmap, 'w') as labelfile:
        for label in labels:
            labelfile.writelines(str(label)+'\n')


if __name__ == '__main__':
    #gen_train_test_sets('/home/wz/DataSets/Offline/CASIA-HWDB1.1/IMG_CLS',
                        # '/home/wz/DataSets/Offline/CASIA-HWDB1.1/labelmap_200.txt',
                        # '/home/wz/DataSets/Offline/CASIA-HWDB1.1/trainval_200.txt',
                        # '/home/wz/DataSets/Offline/CASIA-HWDB1.1/test_200.txt', 200)

    gnt_root_dir = '/home/wz/DataSets/Offline/CASIA-HWDB1.0/Data'
    img_save_dir = '/home/wz/DataSets/Offline/CASIA-HWDB1.0/IMG_CLS'

    #初始化保存文件夹，节省处理时间
    labelmap_ori = ''
    init_save_dir(labelmap_ori, img_save_dir)

    index_dict = {}
    for gnt in os.listdir(gnt_root_dir):
        gnt_path = os.path.join(gnt_root_dir, gnt)
        samples = decode_gnt_to_imgs(gnt_path)
        for sample in samples:
            char_name = sample[0]
            if char_name == '.': char_name = 'dot'   #避免文件夹路径冲突
            if char_name == '/': char_name = 'slash' #避免文件夹路径冲突
            if not char_name in index_dict: index_dict.update({char_name: 0})
            else: index_dict[char_name] += 1
            cv2.imwrite(os.path.join(img_save_dir, char_name, str(index_dict[char_name]) + '.jpg'), sample[1])
        print('processed gnt file %s.' % gnt)