# -*- coding: utf-8 -*-
import codecs

import struct
import numpy as np
import cv2
import os
import random
from pascal_voc_io import *

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

def decode_GNT_to_imgs(gnt):
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

def decode_DGR_to_imgs_and_vocxml(dgr):
    '''
    :param dgr: a writer's encoded ground truth file.
    :return:    samples list, each sample with format (charname, img)
    '''
    doc_img, voc_xml = None, None
    with codecs.open(dgr, mode='rb') as fin:
        while(True):
            left_cache = fin.read(4)
            if len(left_cache) < 4:
                break

            #FILE HEAFER
            size_of_header = struct.unpack("I", left_cache)[0]
            format_code = fin.read(8)
            illus_len = size_of_header - 36
            illus = fin.read(illus_len)
            if sys.version_info < (3, 5):
                code_type = fin.read(20).decode('ASCII')
            else:
                code_type = str(fin.read(20), 'ASCII')

            code_len = struct.unpack("h", fin.read(2))[0]
            bits_per_pix = struct.unpack("h", fin.read(2))[0]

            if bits_per_pix == 1:
                break

            #Image Records (concatenated)
            height = struct.unpack("I", fin.read(4))[0]
            width = struct.unpack("I", fin.read(4))[0]
            doc_img = np.zeros(shape=[height, width], dtype=np.uint8) + 255

            voc_xml = PascalVocWriter(os.path.dirname(dgr), os.path.split(dgr)[-1][:-4] + '.jpg', doc_img.shape)

            # Line Records (concatenated)
            line_num = struct.unpack("I", fin.read(4))[0]
            for i in range(line_num):
                # Character Records (concatenated)
                word_num = struct.unpack("I", fin.read(4))[0]
                for j in range(word_num):
                    tmp_code = fin.read(code_len)
                    try:
                        if sys.version_info < (3, 5):
                            label = tmp_code.decode('gbk')[0]
                        else:
                            label = str(tmp_code, ('gbk'))[0]
                    except:
                        label = u'Unknown'

                    top = struct.unpack("H", fin.read(2))[0]
                    left = struct.unpack("H", fin.read(2))[0]
                    char_height = struct.unpack("H", fin.read(2))[0]
                    char_width = struct.unpack("H", fin.read(2))[0]
                    tmp_img = np.zeros(shape=[char_height, char_width], dtype=np.uint8)

                    #Image data
                    for r in range(char_height):
                        for c in range(char_width):
                            tmp_img[r, c] = struct.unpack("B", fin.read(1))[0]
                    doc_img[top:top+char_height, left:left+char_width] = tmp_img
                    voc_xml.addBndBox(left, top, left + char_width, top+char_height, label)

    return doc_img, voc_xml

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
    samples_train = []
    samples_test = []
    labels = []
    subdirs = os.listdir(img_root_dir)
    subdirs.sort()

    if subsetnum is None:
        subsetnum = len(subdirs)

    for id, folder in enumerate(subdirs[:subsetnum]):
        imgs = os.listdir(os.path.join(img_root_dir, folder))
        for iid, img in enumerate(imgs):
            if iid < 0.8 * len(imgs):
                samples_train.append(folder + '/' + img + ' ' + str(id))
            else:
                samples_test.append(folder + '/' + img + ' ' + str(id))
        # labels.append(folder+','+str(id))
        labels.append(folder)

    random.shuffle(samples_train)
    random.shuffle(samples_test)
    with open(train_index, 'w') as trainset:
        for sample in samples_train:
            trainset.writelines(sample+'\n')

    with open(test_index, 'w') as testset:
        for sample in samples_test:
            testset.writelines(sample+'\n')

    with open(labelmap, 'w') as labelfile:
        for label in labels:
            labelfile.writelines(str(label)+'\n')


if __name__ == '__main__':
    #gen_train_test_sets('/home/wz/DataSets/Offline/CASIA-HWDB1.1/IMG_CLS',
                        # '/home/wz/DataSets/Offline/CASIA-HWDB1.1/labelmap_200.txt',
                        # '/home/wz/DataSets/Offline/CASIA-HWDB1.1/trainval_200.txt',
                        # '/home/wz/DataSets/Offline/CASIA-HWDB1.1/test_200.txt', 200)


    # cv2.imshow('doc_img', doc_img)
    # cv2.waitKey(0)
    '''
    gnt_root_dir = '/home/wz/DataSets/Offline/CASIA-HWDB1.1/Data'
    img_save_dir = '/home/wz/DataSets/Offline/CASIA-HWDB1.1/IMG_CLS'

    #初始化保存文件夹，节省处理时间
    #labelmap_ori = ''
    #init_save_dir(labelmap_ori, img_save_dir)

    index_dict = {}
    for gnt in os.listdir(gnt_root_dir):
        gnt_path = os.path.join(gnt_root_dir, gnt)
        samples = decode_GNT_to_imgs(gnt_path)
        for sample in samples:
            char_name = sample[0]
            if char_name == '.': char_name = 'dot'   #避免文件夹路径冲突
            if char_name == '/': char_name = 'slash' #避免文件夹路径冲突
            if not char_name in index_dict: index_dict.update({char_name: 0})
            else: index_dict[char_name] += 1
            cv2.imwrite(os.path.join(img_save_dir, char_name, str(index_dict[char_name]) + '.jpg'), sample[1])
        print('processed gnt file %s.' % gnt)
    '''


    ##########################2.0##################################
    dgr_data_dir = '/home/wz/DataSets/Offline/CASIA-HWDB2.2/Test_Dgr'
    img_save_dir = '/home/wz/DataSets/Offline/CASIA-HWDB2.2/DOC_IMG/TEST'
    xml_writing_dir = '/home/wz/DataSets/Offline/CASIA-HWDB2.2/DOC_IMG/TEST_XML'
    dgrs = os.listdir(dgr_data_dir)
    for dgr in dgrs[:10]:
        doc_img, voc_xml = decode_DGR_to_imgs_and_vocxml(os.path.join(dgr_data_dir, dgr))
        cv2.imwrite(os.path.join(img_save_dir, dgr[:-4]+'.jpg'), doc_img)
        voc_xml.save(xml_writing_dir + "/" + dgr[:-4] + XML_EXT)
        print('Processed file %s.' % dgr)