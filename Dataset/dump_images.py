#!/usr/bin/env python
# coding: utf-8

# In[38]:


from scipy.io import loadmat
import glob
import cv2
from shutil import copyfile
import os
import numpy as np
import matplotlib.pylab as plt
from skimage.io import imread
from pathlib import Path
import skimage
from skimage import feature, morphology
from matplotlib.pyplot import figure
import matplotlib
from skimage.color import rgb2gray
import copy
import gc
import sys


# In[39]:

bird_labels = {'head':1, 'leye':2, 'reye':3, 'beak':4, 'torso':5, 'neck':6, 'lwing':7, 'rwing':8, 'lleg':9, 'lfoot':10, 'rleg':11, 'rfoot':12, 'tail':13}

cat_labels = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'nose':6, 'torso':7, 'neck':8, 'lfleg':9, 'lfpa':10, 'rfleg':11, 'rfpa':12, 'lbleg':13, 'lbpa':14, 'rbleg':15, 'rbpa':16, 'tail':17}

cow_labels = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'muzzle':6, 'lhorn':7, 'rhorn':8, 'torso':9, 'neck':10, 'lfuleg':11, 'lflleg':12, 'rfuleg':13, 'rflleg':14, 'lbuleg':15, 'lblleg':16, 'rbuleg':17, 'rblleg':18, 'tail':19}

dog_labels = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'nose':6, 'torso':7, 'neck':8, 'lfleg':9, 'lfpa':10, 'rfleg':11, 'rfpa':12, 'lbleg':13, 'lbpa':14, 'rbleg':15, 'rbpa':16, 'tail':17, 'muzzle':18}

horse_labels = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'muzzle':6, 'lfho':7, 'rfho':8, 'torso':9, 'neck':10, 'lfuleg':11, 'lflleg':12, 'rfuleg':13, 'rflleg':14, 'lbuleg':15, 'lblleg':16, 'rbuleg':17, 'rblleg':18, 'tail':19, 'lbho':20, 'rbho':21}

bottle_labels = {'cap':1, 'body':2}

person_labels = {'head':1, 'leye':2,  'reye':3, 'lear':4, 'rear':5, 'lebrow':6, 'rebrow':7,  'nose':8,  'mouth':9,  'hair':10, 'torso':11, 'neck': 12, 'llarm': 13, 'luarm': 14, 'lhand': 15, 'rlarm':16, 'ruarm':17, 'rhand': 18, 'llleg': 19, 'luleg':20, 'lfoot':21, 'rlleg':22, 'ruleg':23, 'rfoot':24}

bus_labels = { 'frontside':1, 'leftside':2, 'rightside':3, 'backside':4, 'roofside':5, 'leftmirror':6, 'rightmirror':7, 'fliplate':8, 'bliplate':9  }
for ii in range(0,10):
    bus_labels['door_{}'.format(ii+1)] = 10+ii
for ii in range(0,10):
    bus_labels['wheel_{}'.format(ii+1)] = 20+ii
for ii in range(0,10):
    bus_labels['headlight_{}'.format(ii+1)] = 30+ii
for ii in range(0,10):
    bus_labels['window_{}'.format(ii+1)] = 40+ii

aeroplane_labels = {'body': 1, 'stern': 2, 'lwing': 3, 'rwing':4, 'tail':5}
for ii in range(0, 10):
    aeroplane_labels['engine_{}'.format(ii+1)] = 6+ii
for ii in range(0, 10):
    aeroplane_labels['wheel_{}'.format(ii+1)] = 16+ii


motorbike_labels = {'fwheel': 1, 'bwheel': 2, 'handlebar': 3, 'saddle': 4}
for ii in range(0,10):
    motorbike_labels['headlight_{}'.format(ii+1)] = 5+ii
motorbike_labels['body']=15

bicycle_labels = {'fwheel': 1, 'bwheel': 2, 'saddle': 3, 'handlebar': 4, 'chainwheel': 5}
for ii in range(0,10):
    bicycle_labels['headlight_{}'.format(ii+1)] = 6+ii
bicycle_labels['body']=16

train_labels = {'head':1,'hfrontside':2,'hleftside':3,'hrightside':4,'hbackside':5,'hroofside':6}
for ii in  range(0,10):
    train_labels['headlight_{}'.format(ii+1)] = 7 + ii
for ii in  range(0,10):
    train_labels['coach_{}'.format(ii+1)] = 17 + ii
for ii in  range(0,10):
    train_labels['cfrontside_{}'.format(ii+1)] = 27 + ii
for ii in  range(0,10):
    train_labels['cleftside_{}'.format(ii+1)] = 37 + ii
for ii in  range(0,10):
    train_labels['crightside_{}'.format(ii+1)] = 47 + ii
for ii in  range(0,10):
    train_labels['cbackside_{}'.format(ii+1)] = 57 + ii
for ii in  range(0,10):
    train_labels['croofside_{}'.format(ii+1)] = 67 + ii

sheep_labels = cow_labels
car_labels = bus_labels

part_labels = {'bird': bird_labels, 'cat': cat_labels, 'cow': cow_labels, 'dog': dog_labels, 'sheep': sheep_labels, 'horse':horse_labels, 'car':car_labels, 'bus':bus_labels, 'bicycle':bicycle_labels, 'motorbike':motorbike_labels, 'person':person_labels,'aeroplane':aeroplane_labels, 'train':train_labels}

# In[40]:


object_name = sys.argv[1]
animals = [object_name]


# In[4]:


def rotate_im(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image


# In[5]:


def get_corners(bboxes):

    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)

    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)

    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))

    return corners


# In[6]:


def clip_box(bbox, clip_box, alpha):
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))

    delta_area = ((ar_ - bbox_area(bbox))/ar_)

    mask = (delta_area < (1 - alpha)).astype(int)

    bbox = bbox[mask == 1,:]


    return bbox


# In[7]:


def rotate_box(corners,angle,  cx, cy, h, w):

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)


    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T

    calculated = calculated.reshape(-1,8)

    return calculated


# In[8]:


def get_enclosing_box(corners):
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]

    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)

    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))

    return final


# In[9]:


def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])


# In[10]:


def rtt(angle, img, bboxes):


    w,h = img.shape[1], img.shape[0]
    cx, cy = w//2, h//2

    img = rotate_im(img, angle)

    corners = get_corners(bboxes)

    corners = np.hstack((corners, bboxes[:,4:]))


    corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)

    new_bbox = get_enclosing_box(corners)


    scale_factor_x = img.shape[1] / w

    scale_factor_y = img.shape[0] / h

    img = cv2.resize(img, (w,h))

    new_bbox[:,:4] = np.true_divide(new_bbox[:,:4], [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y])

    bboxes  = new_bbox

    #bboxes = clip_box(bboxes, [0,0,w, h], 0.25)

    return img, bboxes


# In[11]:


def parts(annopath):
    data = loadmat(annopath)['anno'][0, 0]
    d = {}
    for obj in data['objects'][0, :]:
        p = get_parts(obj)
        bp = {}
        for body_parts in p:
            bp[str(body_parts[0][0])] = body_parts['mask']
        bp['body'] = obj['mask']
        if obj[0][0] in animals:
            d[obj[0][0]] = bp
    return d

# In[12]:


def get_parts(obj):
    name = obj['class'][0]
    index = obj['class_ind'][0, 0]
    n = obj['parts'].shape[1]
    parts = []
    if n > 0:
        for part in obj['parts'][0, :]:
            parts.append(part)
    return parts


# In[13]:


def bounder(img):
    result = np.where(img!=255)
    listOfCoordinates = list(zip(result[0], result[1]))
    for cord in listOfCoordinates:
        img[cord] = 0
    result1 = np.where(img==255)
    listOfCoordinates1 = list(zip(result1[0], result1[1]))
    for cord in listOfCoordinates1:
        img[cord] = 1
    return img


# In[14]:


def cordinates(img):
    y_min = 0
    y_max = 0
    x_min = 0
    x_max = 0

    for i in img:
        if np.count_nonzero(i) is not 0:
            break
        y_min+=1

    for i in img.T:
        if np.count_nonzero(i) is not 0:
            break
        x_min+=1

    for i in img[::-1]:
        if np.count_nonzero(i) is not 0:
            break
        y_max+=1
    y_max = img.shape[0] - y_max - 1

    for i in img.T[::-1]:
        if np.count_nonzero(i) is not 0:
            break
        x_max+=1
    x_max = img.shape[1] - x_max - 1

    return x_min, y_min, x_max, y_max


# In[15]:


def gray(img):
    return rgb2gray(img)


# In[16]:


def edges(img):
    d = morphology.dilation(img, selem=None, out=None, shift_x=False, shift_y=False)
    #d = morphology.dilation(img, selem=None, out=None, shift_x=False, shift_y=False)
    #d = morphology.dilation(img, selem=None, out=None, shift_x=False, shift_y=False)
    e = morphology.erosion(img, selem=None, out=None, shift_x=False, shift_y=False)
    i = d-e
    return i


# In[17]:


def change_size(image, desired_size):

    old_size = image.shape

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    #print(top, bottom, left, right, old_size[0],  old_size[1])

    color = [0,0,0]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


# In[18]:


def label_mask(parts_dic, labels):
    label_mask = 0
    for key, value in parts_dic.items():
        result = np.where(value == 1)
        listOfCoordinates= list(zip(result[0], result[1]))
        for cord in listOfCoordinates:
            value[cord] = labels[key]
        label_mask = label_mask + value
    return label_mask


# In[19]:


def darker(img):

    result = np.where(img!=255)
    listOfCoordinates = list(zip(result[0], result[1]))
    for cord in listOfCoordinates:
        img[cord] = 0
    return img


# In[20]:


def seg_recnstrct(parts_dic, labels):
    seg = {}
    img = 0
    for key, value in parts_dic.items():
        #value = edges(value)
        seg[key]= value
        img = img + value
    img = np.invert(img)
    img = darker(img)
    #label = label_mask(parts_dic, labels)
    #img = skimage.color.label2rgb(label, image=img, colors=None, alpha=0.3, bg_label=-1, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')
    #img = edges(img)
    return img, seg


# In[21]:


def save_sketch_to_dir(img, img_name):
    figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    file_name = './sketches/' + img_name + '.png'
    matplotlib.image.imsave(file_name, img, cmap = 'gray')


# In[23]:


import csv
def animal_list_maker():
    animal_list = {}
    for animal in animals:
        file_name = animal + '_trainval.txt'
        with open('ImageSets/' + file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            line_count = 0
            n = 0
            for row in csv_reader:
                if row[-1] == '1':
                    n+=1
                    annopath = './Annotations_Part/' + row[0] + '.mat'
                    my_file = Path(annopath)
                    if my_file.is_file():
                        animal_list[row[0]] = parts(annopath)
    return animal_list


# In[24]:


def final_dic_images():
    final_dic = {}
    i = 0
    segs = []
    images = {}
    animal_list = animal_list_maker()
    for file_name in animal_list:
        for animal_name in animal_list[file_name]:
            if len(animal_list[file_name][animal_name]) > 0:
                labels = part_labels[animal_name]
                parts_dic = animal_list[file_name][animal_name]
                parts_dic1 = {}
                img, seg = seg_recnstrct(parts_dic, labels)
                segs.append(seg)
                ll = bounder(img)
                ll= 1-ll
                x_min, y_min, x_max, y_max = cordinates(ll)
                h = y_max - y_min
                w = x_max - x_min
                img = img[y_min:y_min+h , x_min:x_min+w]
                img = 1 - img
                img = np.stack((img,)*3, axis=-1)
                #YO YO
                #img111 = np.multiply(img, cv2.imread('./JPEGImages/'+file_name+'.jpg')[y_min:y_min+h , x_min:x_min+w] )
                img111 = cv2.imread('./JPEGImages/'+file_name+'.jpg')[y_min:y_min+h , x_min:x_min+w]
                final_dic[str(i)] = img111
                images[str(i)] = img
                i+=1
    return final_dic, images


print("final dictionary in construction...")
bbx, images = final_dic_images()
print("final dictionary constructed.")


output_list = []
for img in bbx:
    output_list.append(bbx[img])


import pickle
with open(object_name + '_images', 'wb') as f:
    pickle.dump(output_list, f)

print("all images saved!!")

