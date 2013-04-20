#!/usr/bin/env python2
# coding: UTF-8
#
# Python tool for creating average images from faces
# 
# ref:
#     https://github.com/ruoat/averageface
#     http://www.mathworks.cn/matlabcentral/fileexchange/28927
# 
import os
import sys
import glob
import shelve
import numpy as np
import cv2
import cv2.cv as cv
import Image
import ImageFilter

nose_classifier_filename = './haarcascade_mcs_nose.xml'
eyes_classifier_filename = "./haarcascade_eye.xml"
cache_shelve_path = '.cache.shelve'
test_faces = '/tmp/faces/*'
ERROR_FLAG = 'HEHE_ERROR'

def detect_nose(img,cascade=cv2.CascadeClassifier(nose_classifier_filename)):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(2, 2), flags = cv.CV_HAAR_SCALE_IMAGE)
    assert len(rects) >= 1, "fail to detect any noses"
    nosepositions = rects[:,:2] + rects[:,2:] / 2 #center of each nose
    nose_position = nosepositions[0]
    nose_position = np.array(nose_position)
    return nose_position
    #for (x, y) in nosepositions:
    #    cv2.rectangle(img, (x, y), (x, y),(0, 100, 0), 5)
    #cv2.imwrite("/tmp/img.jpg",img)
    #Image.open('/tmp/img.jpg').show()

def detect_eyes(img, cascade=cv2.CascadeClassifier(eyes_classifier_filename)):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(2, 2), flags = cv.CV_HAAR_SCALE_IMAGE)
    assert len(rects) >= 2, "fail to detect any eyes"
    eyepositions = rects[:,:2] + rects[:,2:] / 2 #center of each eye
    eye_pairs = []
    for i, a in enumerate(eyepositions):
        for b in eyepositions[i+1:]:
            if abs(a[1] - b[1]) < 5 and abs(a[0] - b[0]) > 10:
                if a[0] > b[0]:a, b = b, a 
                eye_pairs.append((a, b))
    if len(eye_pairs) == 0:
        raise ValueError, "fail to detect any eyes"
    eyes = max(eye_pairs, key = lambda p: abs(p[0][0] - p[1][0]))
    l, r = eyes
    l = np.array(l)
    r = np.array(r)
    return l, r
    #(lx, ly), (rx, ry) = eyes
    #cv2.rectangle(img, (lx, ly), (lx, ly),(0, 255, 0), 5)
    #cv2.rectangle(img, (rx, ry), (rx, ry),(0, 0, 255), 5)
    #cv2.imwrite("/tmp/img.jpg",img)
    #Image.open('/tmp/img.jpg').show()

def get_image_info(files):
    image_info = []
    fh = shelve.open(cache_shelve_path)
    cache = fh.get('af', {})
    for img_path in files:
        c = cache.get(img_path)
        if c is not None:
            if c != ERROR_FLAG:
                image_info.append(c)
            continue
        try:
            assert os.path.exists(img_path)
            img = cv2.imread(img_path)
            nose_pos = detect_nose(img)
            lefteye_pos, righteye_pos = detect_eyes(img)
            width = img.shape[1]
            height = img.shape[0]

            assert lefteye_pos[0] < nose_pos[0] < righteye_pos[0]
            assert lefteye_pos[1] < nose_pos[1]
            assert righteye_pos[1] < nose_pos[1]
        except:
            cache[img_path] = ERROR_FLAG
            continue
        else:
            info = (img_path, width, height, nose_pos, lefteye_pos, righteye_pos)
            image_info.append(info)
            cache[img_path] = info
    fh['af'] = cache
    fh.close()
    return image_info

def calculate_average(image_info):
    average_width = np.double(0.0)
    average_height = np.double(0.0)
    average_nose_pos = np.double(0.0)
    average_lefteye_pos = np.double(0.0)
    average_righteye_pos = np.double(0.0)

    for i, (img_path, width, height, nose_pos, lefteye_pos, righteye_pos) in enumerate(image_info):
        average_width = 1. * (average_width * i + width) / (i + 1)
        average_height = 1. * (average_height* i + height) / (i + 1)
        average_nose_pos = 1. * (average_nose_pos* i + nose_pos) / (i + 1)
        average_lefteye_pos = 1. * (average_lefteye_pos* i + lefteye_pos) / (i + 1)
        average_righteye_pos= 1. * (average_righteye_pos* i + righteye_pos) / (i + 1)

    average_width = average_width.astype(int)
    average_height = average_height.astype(int)
    average_nose_pos = average_nose_pos.astype(int)
    average_lefteye_pos = average_lefteye_pos.astype(int)
    average_righteye_pos = average_righteye_pos.astype(int)

    return average_width, average_height, average_nose_pos, average_lefteye_pos, average_righteye_pos
   

def test():
    img_path = np.random.choice(glob.glob(test_faces))
    img = cv2.imread(img_path)
    try:
        nose_pos = detect_nose(img)
        lefteye_pos, righteye_pos = detect_eyes(img)
        print nose_pos
        print lefteye_pos, righteye_pos
        assert lefteye_pos[0] < nose_pos[0] < righteye_pos[0]
        assert lefteye_pos[1] < nose_pos[1]
        assert righteye_pos[1] < nose_pos[1]
    except Exception, e:
        print e
        pass
    else:
        nose_pos = tuple(nose_pos)
        lefteye_pos = tuple(lefteye_pos)
        righteye_pos= tuple(righteye_pos)
        cv2.rectangle(img, nose_pos, nose_pos,(0, 100, 0), 5)
        cv2.rectangle(img, lefteye_pos, lefteye_pos,(0, 100, 0), 5)
        cv2.rectangle(img, righteye_pos, righteye_pos,(0, 100, 0), 5)
        cv2.imwrite("/tmp/img.jpg",img)
        Image.open('/tmp/img.jpg').show()



def main(argv):
    if len(argv) != 3:
        print 'Usage: python %s "/path/support/wildcard/foo*bar.png" output.png' % (argv[0])
        sys.exit()

    print 'extract eyes features and nose features ...'
    pathname = argv[1]
    files = glob.glob(pathname)
    image_info = get_image_info(files)
    if len(image_info) == 0:
        print '0 image exists in %s' % argv[1]
        sys.exit()

    average_width, average_height, average_nose_pos, average_lefteye_pos, average_righteye_pos = calculate_average(image_info)
    average_eye_width = average_righteye_pos[0] - average_lefteye_pos[0]
    average_nose_height = average_nose_pos[1] - (average_lefteye_pos[1] + average_righteye_pos[1]) / 2

    average_image = np.empty([average_height, average_width, 3])
    print 'generating average face using %s images...' % len(image_info) 
    for i, (img_path, width, height, nose_pos, lefteye_pos, righteye_pos) in enumerate(image_info):
        eye_width = righteye_pos[0] - lefteye_pos[0]
        nose_height = nose_pos[1] - (lefteye_pos[1] + righteye_pos[1]) / 2
        resize_factor = (average_eye_width * 1.0 / eye_width, average_nose_height * 1.0 / nose_height)
        img = cv2.imread(img_path)
        if resize_factor != (1.0, 1.0):
            width = int(round(width * resize_factor[0]))
            height = int(round(height * resize_factor[1]))
            nose_pos = (nose_pos * resize_factor).astype(int)
            lefteye_pos = (lefteye_pos * resize_factor).astype(int)
            righteye_pos = (righteye_pos * resize_factor).astype(int)
            img = cv2.resize(img, (width, height))

        x_diff = average_lefteye_pos[0] - lefteye_pos[0]
        y_diff = average_lefteye_pos[1] - lefteye_pos[1]


        if (x_diff >= 0):
            source_x_offset = 0
            target_x_offset = x_diff
        else:
            source_x_offset = abs(x_diff)
            target_x_offset = 0
        x_size = min(average_width - target_x_offset, width - source_x_offset)

        if (y_diff >= 0):
            source_y_offset = 0
            target_y_offset = y_diff
        else:
            source_y_offset = abs(y_diff)
            target_y_offset = 0
        y_size = min(average_height - target_y_offset, height - source_y_offset)

        temp_image = np.empty([average_height, average_width,3])
        temp_image.fill(255)
        
        temp_image[target_y_offset:target_y_offset+y_size, target_x_offset:target_x_offset+x_size] = \
                img[source_y_offset:source_y_offset+y_size, source_x_offset:source_x_offset+x_size]
    
        average_image = 1. * (average_image * i + temp_image) / (i + 1)
    average_image = average_image.astype(int)

    af_path = argv[2]
    cv2.imwrite(af_path, average_image)
    img = Image.open(af_path)
    img = img.filter(ImageFilter.MedianFilter(3))
    img.save(af_path)
    img.show()

    
if __name__ == '__main__':
    main(sys.argv)
