import sys
sys.path.insert(0, '/home/wz/DeepLearning/caffe_dir/easy-pvanet/caffe-fast-rcnn/python')#py2
# sys.path.insert(0, '/home/wz/DeepLearning/caffe_dir/caffe/python') #py3
import caffe
import cv2
import numpy as np
import os
import codecs

CLASSIFIER_MODEL_FILE = '/home/wz/PycharmProjects/HWDB/model/googlenet-deploy.prototxt'
CLASSIFIER_PRETRAINED = '/home/wz/PycharmProjects/HWDB/model/googlenet.caffemodel'
caffe.set_mode_gpu()
goolenet_classifier = caffe.Classifier(CLASSIFIER_MODEL_FILE, CLASSIFIER_PRETRAINED)


with codecs.open('/home/wz/DataSets/Offline/CASIA-HWDB1.1/labelmap.txt', encoding='utf-8', mode='r') as labelfile:
    label_lines = labelfile.readlines()
    print(len(label_lines))

testset = '/home/wz/DataSets/Offline/CASIA-HWDB1.1/test_all.txt'
with open(testset, 'r') as testfile:
    lines = testfile.readlines()
    for line in lines[:10]:
        IMAGE_FILE = os.path.join('/home/wz/DataSets/Offline/CASIA-HWDB1.1/IMG_CLS/', line.split(' ')[0])
        show = cv2.imread(IMAGE_FILE, cv2.IMREAD_COLOR)
        input_array = np.zeros(shape = (show.shape[0], show.shape[1], 3), dtype=np.float)
        input_array[:, :, :] = show
        input_array[:, :, 0] -= 104.0
        input_array[:, :, 1] -= 117.0
        input_array[:, :, 2] -= 123.0

        prediction = goolenet_classifier.predict([input_array])[0]
        top_1_index = prediction.argmax()
        print(label_lines[top_1_index])