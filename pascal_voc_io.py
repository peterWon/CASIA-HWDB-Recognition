#!/usr/bin/env python
# -*- coding: utf8 -*-
#import _init_path
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

XML_EXT = '.xml'


class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def renameFileName(self,filename):
        self.filename = filename

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf-8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        top.set('verified', 'yes' if self.verified else 'no')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        localImgPath = SubElement(top, 'path')
        localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax, self.imgSize[1] - 1)
        ymax = min(ymax, self.imgSize[0] - 1)

        # 判断是否box是否是边界处的狭长box
        if abs((xmax - xmin) * 1.0 / (ymax - ymin)) < 0.25 and (xmin < 0.1 * self.imgSize[1] or xmax > 0.9 * self.imgSize[1]):
            return None
        if abs((xmax - xmin) * 1.0 / (ymax - ymin)) > 4 and (ymin < 0.1 * self.imgSize[0] or ymax > 0.9 * self.imgSize[0]):
            return None

        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            try:
                name.text = unicode(each_object['name'])
            except NameError:
                # Py3: NameError: name 'unicode' is not defined
                name.text = each_object['name']
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = "0"
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding='utf-8')
        else:
            out_file = codecs.open(targetFile, 'w', encoding='utf-8')

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class PascalVocReader:
    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color]
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        self.width = 0
        self.height = 0

        self.parseXML()

    def getShapes(self):
        return self.shapes

    def addShape(self, label, bndbox):
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None))

    def parseXML(self):
        assert self.filepath.endswith('.xml'), "Unsupport file format"
        parser = etree.XMLParser(encoding='utf-8')
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        size = xmltree.find('size')
        self.width = int(size.find('width').text)
        self.height = int(size.find('height').text)
        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            #wz, check if bbox out of range.
            '''
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            if ymin < 0 or ymax > height or xmin < 0 or xmax > width:
                self.hasExceedBoundingBBox = True
            '''

            label = object_iter.find('name').text
            self.addShape(label, bndbox)
        return True

