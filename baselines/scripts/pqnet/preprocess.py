import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
import pickle
import math
from tensorflow.keras import backend as K
import sys
import cv2
import psutil
from skimage.transform import resize
max_num_node = 24
canvas_size = 550

flip_bird = [1, 3, 2, 4, 5, 6, 10, 7, 11, 12, 9, 10, 13]
flip_cow = [1, 3, 2, 5, 4, 6, 10, 7, 9, 10, 13, 14, 11, 12, 17, 110, 15, 16, 19]
flip_cat = [1, 3, 2, 5, 4, 6, 7, 8, 11, 12, 9, 10, 15, 16, 13, 14, 17]
flip_dog = [1, 3, 2, 5, 4, 6, 7, 8, 11, 12, 9, 10, 15, 16, 13, 14, 17, 18]
flip_horse = [1, 3, 2, 5, 4, 6, 8, 7, 9, 10, 13, 14, 11, 12, 17, 18, 15, 16, 19, 21, 20]
flip_person = [1, 3, 2, 5, 4, 7, 6, 8, 9, 10, 11, 12, 16, 17, 18, 13, 14, 15, 22, 23, 24, 19, 20, 21]
flip_sheep = flip_cow

person_tree = {}
person_tree[0] = [1, 2, 3, 4, 7, 8, 9, 10, 11]
person_tree[1] = [0, 2, 3, 5, 7]
person_tree[2] = [0, 1, 4, 6, 7]
person_tree[3] = [0, 1]
person_tree[4] = [0, 2]
person_tree[5] = [1]
person_tree[6] = [2]
person_tree[7] = [0, 1, 2, 8]
person_tree[8] = [0, 7]
person_tree[9] = [0]
person_tree[10] = [0, 11, 13, 16, 19, 22]
person_tree[11] = [0, 10]
person_tree[12] = [13, 14]
person_tree[13] = [10, 12]
person_tree[14] = [12]
person_tree[15] = [16, 17]
person_tree[16] = [10, 15]
person_tree[17] = [15]
person_tree[18] = [19, 20]
person_tree[19] = [18, 10]
person_tree[20] = [18]
person_tree[21] = [22, 23]
person_tree[22] = [21, 10]
person_tree[23] = [21]

bird_tree = {}
bird_tree[0] = [1, 2, 3, 4, 5]
bird_tree[1] = [0, 2, 3]
bird_tree[2] = [0, 1, 3]
bird_tree[3] = [0, 1, 2]
bird_tree[4] = [0, 5, 6, 7, 8, 10, 12]
bird_tree[5] = [0, 4]
bird_tree[6] = [7, 4]
bird_tree[7] = [6, 4]
bird_tree[8] = [4, 9]
bird_tree[9] = [8]
bird_tree[10] = [11, 4]
bird_tree[11] = [10]
bird_tree[12] = [4]

dog_tree = {}
dog_tree[0] = [1, 2, 3, 4, 5, 6, 7, 17]
dog_tree[1] = [2, 0, 3]
dog_tree[2] = [0, 1, 4]
dog_tree[3] = [0, 1]
dog_tree[4] = [0, 2]
dog_tree[5] = [0, 1, 2]
dog_tree[6] = [0, 8, 10, 12, 14, 16, 7]
dog_tree[7] = [0, 6]
dog_tree[8] = [9, 6]
dog_tree[9] = [8]
dog_tree[10] = [6, 11]
dog_tree[11] = [10]
dog_tree[12] = [13, 6]
dog_tree[13] = [12]
dog_tree[14] = [6, 15]
dog_tree[15] = [14]
dog_tree[16] = [6]
dog_tree[17] = [0]

cat_tree = {}
cat_tree[0] = [1, 2, 3, 4, 5, 6, 7]
cat_tree[1] = [2, 0, 3]
cat_tree[2] = [0, 1, 4]
cat_tree[3] = [0, 1]
cat_tree[4] = [0, 2]
cat_tree[5] = [0, 1, 2]
cat_tree[6] = [0, 8, 10, 12, 14, 16, 7]
cat_tree[7] = [0, 6]
cat_tree[8] = [9, 6]
cat_tree[9] = [8]
cat_tree[10] = [6, 11]
cat_tree[11] = [10]
cat_tree[12] = [13, 6]
cat_tree[13] = [12]
cat_tree[14] = [6, 15]
cat_tree[15] = [14]
cat_tree[16] = [6]

horse_tree = {}
horse_tree[0] = [1, 2, 3, 4, 5, 8, 9]
horse_tree[1] = [2, 0, 3]
horse_tree[2] = [0, 1, 4]
horse_tree[3] = [0, 1]
horse_tree[4] = [0, 2]
horse_tree[5] = [0, 1, 2]
horse_tree[6] = [11]  # lfho
horse_tree[7] = [13]  # rfho
horse_tree[8] = [0, 10, 12, 14, 16, 18]
horse_tree[9] = [0, 8]
horse_tree[10] = [8, 11, 12]
horse_tree[11] = [10, 6]
horse_tree[12] = [10, 8, 13]
horse_tree[13] = [7]
horse_tree[14] = [8, 15, 16]
horse_tree[15] = [14, 19]
horse_tree[16] = [14, 17]
horse_tree[17] = [16, 20]
horse_tree[18] = [8]
horse_tree[19] = [15]
horse_tree[20] = [17]

cow_tree = {}
cow_tree[0] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
cow_tree[1] = [2, 0, 3, 5]
cow_tree[2] = [0, 1, 4, 5]
cow_tree[3] = [0, 1, 6]
cow_tree[4] = [0, 2, 7]
cow_tree[5] = [0, 1, 2]
cow_tree[6] = [0, 3]  # lfho
cow_tree[7] = [0, 4, 13]  # rfho
cow_tree[8] = [0, 9, 10, 12, 14, 16, 18]
cow_tree[9] = [0, 8]
cow_tree[10] = [8, 11, 12]
cow_tree[11] = [10, 6]
cow_tree[12] = [10, 8, 13]
cow_tree[13] = [7, 12]
cow_tree[14] = [8, 15, 16]
cow_tree[15] = [14]
cow_tree[16] = [8, 14, 17]
cow_tree[17] = [16]
cow_tree[18] = [8]

motorbike_tree = {}
motorbike_tree[0] = [14, 1, 2]
motorbike_tree[1] = [14, 0]
motorbike_tree[2] = [14, 0]
motorbike_tree[3] = [14]
motorbike_tree[4] = [14]
motorbike_tree[5] = [14]
motorbike_tree[6] = [14]
motorbike_tree[7] = [14]
motorbike_tree[8] = [14]
motorbike_tree[9] = [14]
motorbike_tree[10] = [14]
motorbike_tree[11] = [14]
motorbike_tree[12] = [14]
motorbike_tree[13] = [14]
motorbike_tree[14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

bicycle_tree = {}
bicycle_tree[0] = [15, 1, 3]
bicycle_tree[1] = [15, 0, 2, 4]
bicycle_tree[2] = [15, 1]
bicycle_tree[3] = [15, 0]
bicycle_tree[4] = [15, 1]
bicycle_tree[5] = [15]
bicycle_tree[6] = [15]
bicycle_tree[7] = [15]
bicycle_tree[8] = [15]
bicycle_tree[9] = [15]
bicycle_tree[10] = [15]
bicycle_tree[11] = [15]
bicycle_tree[12] = [15]
bicycle_tree[13] = [15]
bicycle_tree[14] = [15]
bicycle_tree[15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

aeroplane_tree = {}
aeroplane_tree[0] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
aeroplane_tree[1] = [0]
aeroplane_tree[2] = [0, 3]
aeroplane_tree[3] = [0, 2]
aeroplane_tree[4] = [0]
aeroplane_tree[5] = [0]
aeroplane_tree[6] = [0]
aeroplane_tree[7] = [0]
aeroplane_tree[8] = [0]
aeroplane_tree[9] = [0]
aeroplane_tree[10] = [0]
aeroplane_tree[11] = [0]
aeroplane_tree[12] = [0]
aeroplane_tree[13] = [0]
aeroplane_tree[14] = [0]
aeroplane_tree[15] = [0]
aeroplane_tree[16] = [0]
aeroplane_tree[17] = [0]
aeroplane_tree[18] = [0]
aeroplane_tree[19] = [0]
aeroplane_tree[20] = [0]
aeroplane_tree[21] = [0]
aeroplane_tree[22] = [0]

tree = {'aeroplane': aeroplane_tree, 'motorbike': motorbike_tree, 'bicycle': bicycle_tree, 'person': person_tree,
        'cow': cow_tree, 'dog': dog_tree, 'cat': cat_tree, 'sheep': cow_tree, 'bird': bird_tree, 'horse': horse_tree}

object_names = ['cow', 'sheep', 'bird', 'person', 'cat', 'dog', 'horse', 'aeroplane', 'motorbike', 'bicycle']

class_dic = {'cow': 0, 'sheep': 1, 'bird': 2, 'person': 3, 'cat': 4, 'dog': 5, 'horse': 6, 'aeroplane': 7,
             'motorbike': 8, 'bicycle': 9, 'car': 10}


def get_pos(bx):
    temp_pos = []
    for i in bx:
        if i.tolist() != [0, 0, 0, 0]:
            temp_pos.append([1])
        elif i.tolist() == [0, 0, 0, 0]:
            temp_pos.append([0])

    return np.asarray(temp_pos)


colors = [(229, 184, 135), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255),
          (130, 0, 75), (0, 128, 128), (128, 128, 0), (128, 128, 128), (0, 0, 0), (30, 105, 210),
          (30, 105 // 2, 210 // 2), (180, 105, 255), (180 // 2, 105 // 2, 255), (100, 100, 30), (0, 100 // 2, 20),
          (128, 0, 128), (30, 105, 210), (255 // 2, 105, 255), (180 // 2, 105, 255 // 2), (50, 100, 0),
          (229 // 2, 184, 135 // 2), (229, 184, 135), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
          (255, 255, 0), (255, 0, 255), (130, 0, 75), (0, 128, 128), (128, 128, 0), (128, 128, 128), (0, 0, 0),
          (30, 105, 210), (30, 105 // 2, 210 // 2), (180, 105, 255), (180 // 2, 105 // 2, 255), (100, 100, 30),
          (0, 100 // 2, 20), (128, 0, 128), (30, 105, 210), (255 // 2, 105, 255), (180 // 2, 105, 255 // 2),
          (50, 100, 0), (229 // 2, 184, 135 // 2)]


def arrangement(a, b, object_name):
    if object_name == 'cow' or object_name == 'sheep':
        p = [10, 11, 18, 13, 12, 14, 16, 15, 17, 9, 0, 7, 3, 4, 5, 6, 1, 2, 8]
    elif object_name == 'bird':
        p = [10, 11, 12, 9, 0, 7, 3, 4, 5, 6, 1, 2, 8]
    elif object_name == 'person':
        p = [10, 11, 19, 18, 20, 22, 21, 23, 13, 12, 14, 16, 15, 17, 9, 0, 7, 3, 4, 5, 6, 1, 2, 8]
    elif object_name == 'cat':
        p = [10, 11, 13, 12, 14, 16, 15, 9, 0, 7, 3, 4, 5, 6, 1, 2, 8]
    elif object_name == 'dog':
        p = [10, 11, 13, 12, 14, 16, 15, 17, 9, 0, 7, 3, 4, 5, 6, 1, 2, 8]
    elif object_name == 'horse':
        p = [10, 11, 19, 18, 20, 13, 12, 14, 16, 15, 17, 9, 0, 7, 3, 4, 5, 6, 1, 2, 8]
    elif object_name == 'aeroplane':
        p = [10, 11, 19, 18, 20, 22, 21, 13, 12, 14, 16, 15, 17, 9, 0, 7, 3, 4, 5, 6, 1, 2, 8]
    elif object_name == 'car':
        p = [10, 11, 19, 18, 20, 22, 21, 23, 24, 25, 26, 27, 28, 13, 12, 14, 16, 15, 17, 9, 0, 7, 3, 4, 5, 6, 1, 2, 8]
    elif object_name == 'motorbike':
        p = [10, 11, 13, 12, 14, 9, 0, 7, 3, 4, 5, 6, 1, 2, 8]
    elif object_name == 'bicycle':
        p = [10, 11, 13, 12, 14, 15, 9, 0, 7, 3, 4, 5, 6, 1, 2, 8]
    else:
        print("error")
    return a[p], b[p]


def rearrange(lbl, bx, mask, object_name):
    if object_name == 'cow' or object_name == 'sheep':
        p = np.asarray([1, 3, 2, 5, 4, 6, 8, 7, 9, 10, 13, 14, 11, 12, 17, 18, 15, 16, 19]) - 1
    elif object_name == 'bird':
        p = np.asarray([1, 3, 2, 4, 5, 6, 8, 7, 11, 12, 9, 10, 13]) - 1
    elif object_name == 'person':
        p = np.asarray([1, 3, 2, 5, 4, 7, 6, 8, 9, 10, 11, 12, 16, 17, 18, 13, 14, 15, 22, 23, 24, 19, 20, 21]) - 1
    elif object_name == 'cat':
        p = np.asarray([1, 3, 2, 5, 4, 6, 7, 8, 11, 12, 9, 10, 15, 16, 13, 14, 17]) - 1
    elif object_name == 'dog':
        p = np.asarray([1, 3, 2, 5, 4, 6, 7, 8, 11, 12, 9, 10, 15, 16, 13, 14, 17, 18]) - 1
    elif object_name == 'horse':
        p = np.asarray([1, 3, 2, 5, 4, 6, 8, 7, 9, 10, 13, 14, 11, 12, 17, 18, 15, 16, 19, 21, 20]) - 1
    elif object_name == 'aeroplane':
        p = np.asarray([1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]) - 1
    elif object_name == 'car':
        p = np.asarray(
            [1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29]) - 1
    elif object_name == 'motorbike':
        p = np.asarray([1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13, 14, 15]) - 1
    elif object_name == 'bicycle':
        p = np.asarray([1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]) - 1
    else:
        print("error")
    return lbl[p], bx[p], mask[p]


def pad_along_axis(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b


def bounder(img):
    result = np.where(img < 0.5)
    listOfCoordinates = list(zip(result[0], result[1]))
    for cord in listOfCoordinates:
        img[cord] = 0
    result1 = np.where(img >= 0.5)
    listOfCoordinates1 = list(zip(result1[0], result1[1]))
    for cord in listOfCoordinates1:
        img[cord] = 1
    return img


def clip_rgb(img):
    result = np.where(img < 0.2)
    listOfCoordinates = list(zip(result[0], result[1], result[2]))
    for cord in listOfCoordinates:
        img[cord] = 0
    return img

def add_images(canvas, img, ii):
    result = np.where(img != 0)
    listOfCoordinates = list(zip(result[0], result[1]))
    for cord in listOfCoordinates:
        canvas[cord] = ii
    return canvas

def add_images_rgb(canvas, img):
    #print("Canvas : " + str(canvas.shape))
    #print("Image :" + str(img.shape))
    #print(img)
    result = np.where(img != 0)
    listOfCoordinates = list(zip(result[0], result[1], result[2]))
    #print(len(listOfCoordinates))
    for cord in listOfCoordinates:
        canvas[cord] = img[cord]
    return canvas


label_to_color = {0: (1, 1, 1),
                  1: (0.941, 0.973, 1),
                  2: (0.98, 0.922, 0.843),
                  3: (0, 1, 1),
                  4: (0.498, 1, 0.831),
                  5: (0.941, 1, 1),
                  6: (0.961, 0.961, 0.863),
                  7: (1, 0.894, 0.769),
                  8: (0.251, 0.878, 0.816),
                  9: (1, 0.388, 0.278),
                  10: (0, 0, 1),
                  11: (0.541, 0.169, 0.886),
                  12: (0.647, 0.165, 0.165),
                  13: (0.871, 0.722, 0.529),
                  14: (0.373, 0.62, 0.627),
                  15: (0.498, 1, 0),
                  16: (0.824, 0.412, 0.118),
                  17: (1, 0.498, 0.314),
                  18: (0.392, 0.584, 0.929),
                  19: (0.275, 0.51, 0.706),
                  20: (0.863, 0.0784, 0.235),
                  21: (0, 1, 1),
                  22: (0, 0, 0.545),
                  23: (0.824, 0.706, 0.549),
                  24: (0.251, 0.878, 0.816)}


def label_2_image(img):
    rgb_img = np.zeros((img.shape[0], img.shape[1], 3))
    for key in label_to_color.keys():
        rgb_img[img == key] = label_to_color[key]
    return rgb_img


def make_rgb_mask(mask, box):
    box = box * canvas_size
    max_parts = box.shape[0]
    #print(box.shape)
    #print(mask.shape)
    xmax = int(np.max(box[:, 2]))
    ymax = int(np.max(box[:, 3]))
    w = xmax
    h = ymax
    #print(xmax)
    #print(ymax)
    canvas = np.zeros((int(ymax), int(xmax), 3), np.float32)
    # b_in, mx = arrangement(b_in, mx,object_name)
    for i in range(max_parts):
        new_mask = mask[i] 
        #cv2.imwrite('mask_output' + str(i) + '.jpg', mask[i]*255)
        x_min, y_min, x_max, y_max = box[i][0], box[i][1], box[i][2], box[i][3]
        x_min = 0 if x_min < 0 else x_min
        y_min = 0 if y_min < 0 else y_min
        if x_max - x_min > 0 and y_max - y_min > 0:
            x, y, c = canvas[int(y_min):int(y_max), int(x_min):int(x_max),:].shape
            print(x)
            print(y)
            print(c)
            if x==0:
              print("{} {} {} {}".format(y_min,y_max, x_min, x_max))
           # print(canvas[int(y_min):int(y_max), int(x_min):int(x_max)].shape)
            if x > 0 and y > 0:
            #print(mask[i].shape)
              img = cv2.resize(clip_rgb(new_mask.copy()), dsize=(y, x))
              #print(img.shape)
              canvas[int(y_min):int(y_max), int(x_min):int(x_max)] = add_images_rgb(
                canvas[int(y_min):int(y_max), int(x_min):int(x_max)],
                img)

    #plt.imshow(canvas)
    #plt.savefig('test.jpg')
    #plt.show()

    # plt.imshow(label_2_image(canvas))
    # plt.show()
    # return label_2_image(canvas)
    return canvas


def make_mask(box, mask):
    b_in = np.copy(box)
    mx = np.copy(mask)
    max_parts = len(box)
    xmax = max(box[:, 2])
    ymax = max(box[:, 3])
    canvas = np.zeros((int(ymax), int(xmax)), np.float32)
    b_in, mx = arrangement(b_in, mx, object_name)
    for i in range(max_parts):
        x_min, y_min, x_max, y_max = b_in[i]
        if x_max - x_min > 0 and y_max - y_min > 0:
            x, y = canvas[int(y_min):int(y_max), int(x_min):int(x_max)].shape
            canvas[int(y_min):int(y_max), int(x_min):int(x_max)] = add_images(
                canvas[int(y_min):int(y_max), int(x_min):int(x_max)],
                cv2.resize(bounder(np.squeeze(mx[i])) * (i + 1), (y, x)), i + 1)
    plt.imshow(label_2_image(canvas))
    plt.show()
    return label_2_image(canvas)


def plot_image_bbx(bbx, image):
    canvas = np.copy(image)
    i = 0
    for coord in bbx:
        x_minp, y_minp, x_maxp, y_maxp = coord
        if [x_minp, y_minp, x_maxp, y_maxp] != [0, 0, 0, 0]:
            cv2.rectangle(canvas, ((x_minp), (y_minp)), ((x_maxp), (y_maxp)), colors[i], 4)
        i = i + 1
    plt.imshow(canvas)
    plt.show()
    return canvas


def flip_mask(mask):
    mx = np.copy(mask)
    for i in range(len(mx)):
        mx[i] = mx[i][:, ::-1]
    return mx


def flip_bbx(label, bbx, img):
    bx = np.copy(bbx)
    x_min = min(bbx[:, 0])
    y_min = min(bbx[:, 1])
    x_max = max(bbx[:, 2])
    y_max = max(bbx[:, 3])
    img_center = np.asarray([((x_max + x_min) / 2), ((y_max + y_min) / 2)])
    img_center = np.hstack((img_center, img_center))
    bx[:, [0, 2]] += 2 * (img_center[[0, 2]] - bx[:, [0, 2]])
    box_w = abs(bx[:, 0] - bx[:, 2])
    bx[:, 0] -= box_w
    bx[:, 2] += box_w
    for i in range(len(label)):
        if sum(label[i]) == 0:
            bx[i][0] = 0
            bx[i][1] = 0
            bx[i][2] = 0
            bx[i][3] = 0
    return bx


def flip_data_instance(label, box, mask, image, object_name):
    bx = np.copy(flip_bbx(label, box, image))
    mx = np.copy(flip_mask(mask))
    ix = np.copy(image[:, ::-1])
    lx = np.copy(label)
    lx, bx, mx = rearrange(lx, bx, mx, object_name)
    return lx, bx, mx, ix


def cordinates(img):
    y_min = 0
    y_max = 0
    x_min = 0
    x_max = 0

    for i in img:
        if np.count_nonzero(i) is not 0:
            break
        y_min += 1

    for i in img.T:
        if np.count_nonzero(i) is not 0:
            break
        x_min += 1

    for i in img[::-1]:
        if np.count_nonzero(i) is not 0:
            break
        y_max += 1
    y_max = img.shape[0] - y_max - 1

    for i in img.T[::-1]:
        if np.count_nonzero(i) is not 0:
            break
        x_max += 1
    x_max = img.shape[1] - x_max - 1

    return x_min, y_min, x_max, y_max


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


def get_corners(bboxes):
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def clip_box(bbox, clip_box, alpha):
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

    delta_area = ((ar_ - bbox_area(bbox)) / ar_)

    mask = (delta_area < (1 - alpha)).astype(int)

    bbox = bbox[mask == 1, :]

    return bbox


def rotate_box(corners, angle, cx, cy, h, w):
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners):
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def rtt(angle, label, img, bboxes):
    w, h = img.shape[1], img.shape[0]
    cx, cy = w // 2, h // 2

    img = rotate_im(img, angle)

    corners = get_corners(bboxes)

    corners = np.hstack((corners, bboxes[:, 4:]))

    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

    new_bbox = get_enclosing_box(corners)

    scale_factor_x = img.shape[1] / w

    scale_factor_y = img.shape[0] / h

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    new_bbox[:, :4] = np.true_divide(new_bbox[:, :4], [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y])

    for i in range(len(label)):
        if (label[i]) == 0:
            new_bbox[i][0] = 0
            new_bbox[i][1] = 0
            new_bbox[i][2] = 0
            new_bbox[i][3] = 0

    return img, new_bbox


def render_mask(box, mask, angle):
    mx = np.copy(mask)
    b_in = np.copy(box)
    max_parts = len(box)
    xmax = max(box[:, 2])
    ymax = max(box[:, 3])
    temp_mx_list = []
    temp_bx_list = []
    for i in range(max_parts):
        canvas = np.zeros((int(ymax), int(xmax)), np.float32)
        x_min, y_min, x_max, y_max = b_in[i]
        if x_max - x_min > 0 and y_max - y_min > 0:
            x, y = canvas[int(y_min):int(y_max), int(x_min):int(x_max)].shape
            canvas[int(y_min):int(y_max), int(x_min):int(x_max)] = add_images(
                canvas[int(y_min):int(y_max), int(x_min):int(x_max)],
                cv2.resize(bounder(np.squeeze(mx[i])) * (i + 1), (y, x)), i + 1)
            canvas = rotate_im(canvas, angle)
            x_min, y_min, x_max, y_max = cordinates(canvas)
            # canvas = canvas[int(y_min):int(y_max), int(x_min):int(x_max)]
            # resized_cropped = np.expand_dims(cv2.resize(canvas, (64, 64)), axis = 3)
        temp_bx_list.append([x_min, y_min, x_max, y_max])
        # temp_mx_list.append(resized_cropped)
        # plt.imshow(canvas)
        # plt.show()
    return np.asarray(temp_bx_list, dtype="float32")


def scale(bbx, scaling_factor):
    height = max(bbx[:, 3])
    width = max(bbx[:, 2])

    pos = get_pos(bbx)

    fold_a = np.copy(bbx)
    fold_b = np.copy(bbx)
    fold_c = np.copy(bbx)
    fold_d = np.copy(bbx)

    scale_height = scaling_factor
    scale_width = scaling_factor

    fold_a[:, 0] = (fold_a[:, 0] - scale_width)
    fold_b[:, 1] = (fold_b[:, 1] - scale_height)
    fold_c[:, 2] = (fold_c[:, 2] + scale_width)
    fold_d[:, 3] = (fold_d[:, 3] + scale_height)

    return fold_a * pos, fold_b * pos, fold_c * pos, fold_d * pos


def centre_object(bbx, canvas_size):
    pos = get_pos(bbx)
    bx = np.copy(bbx)

    h, w = canvas_size

    h_o = max(bbx[:, 3])
    w_o = max(bbx[:, 2])

    h_shift = int(h / 2 - h_o / 2)
    w_shift = int(w / 2 - w_o / 2)

    bx[:, 0] = (bx[:, 0] + w_shift)
    bx[:, 1] = (bx[:, 1] + h_shift)
    bx[:, 2] = (bx[:, 2] + w_shift)
    bx[:, 3] = (bx[:, 3] + h_shift)

    return bx * pos


def append_labels(box):
    all_box = []
    for bbx in box:
        pos = get_pos(bbx)
        bbx = (((bbx / canvas_size))) * pos

        temp = []
        for bx in bbx:
            if bx.tolist() != [0, 0, 0, 0]:
                temp.append([1] + bx.tolist())
            else:
                temp.append([0] + bx.tolist())
        all_box.append(temp)
    return np.asarray(all_box)


def plot_bbx(bbx):
    bbx = bbx * canvas_size
    xmax = int(np.max(bbx[:, 2]))
    ymax = int(np.max(bbx[:, 3]))
    w = xmax
    h = ymax
    canvas = np.ones((ymax, xmax, 3), np.float32) * 255.0
    bbx = bbx.tolist()
    i = 0
    for coord in bbx:
        x_minp, y_minp, x_maxp, y_maxp = coord 
        if [x_minp, y_minp, x_maxp, y_maxp] != [0, 0, 0, 0]:
            cv2.rectangle(canvas, (int(x_minp), int(y_minp)), (int(x_maxp), int(y_maxp)), colors[i], 4)
        i = i + 1
    return canvas


def transform_bbx(bbx1):
    eps = 0.00001
    bbx = np.copy(bbx1)
    bxx = np.copy(bbx)

    bbx[:, 0] = np.exp(bbx[:, 0])
    bbx[:, 1] = np.exp(bbx[:, 1])
    bbx[:, 2] = np.exp(bbx[:, 2])
    bbx[:, 3] = np.exp(bbx[:, 3])

    bxx[:, 0] = bbx[:, 0]
    bxx[:, 1] = bbx[:, 1]
    bxx[:, 2] = bbx[:, 0] + (bbx[:, 3])
    bxx[:, 3] = bbx[:, 1] + (bbx[:, 2])

    return bxx


def make_full_mask_list(mask_list, size_of_list):
    ml = []
    for mask in mask_list:
        masks = [np.zeros((64, 64, 1))] * size_of_list
        masks[0:len(mask)] = mask
        ml.append(np.asarray(masks))

    return np.asarray(ml)

    class_v = {}


def get_rgbmask(box, mask, images, object_name):
    """
    box: Nx24x5 bounding boxes
    mask: Nx24x64x64x1 part separated masks
    images: List of N rgb images
    """
    print(box.shape[0])
    print(len(images))

    box = box.astype('float32')

    cropped = []  # list of part separated rgb for all objects: Nx24x64x64x3

    for idx, img in enumerate(images):

        images[idx] = img.astype('uint8')  # change to uint8 due to memory issues

        mmx = mask[idx, :, :, :, 0]  # full-mask of object at position idx

        bbx = box[idx, :, 1:] * canvas_size  # xmin,ymin,xmax,ymax format
        avail = box[idx, :, 0] == 1  # part availability

        bbx = (np.round(bbx)).astype('int')

        h, w = img.shape[:2]
        parts = []  # list of part separated rgb for one object: 24x64x64x3

        for j in range(24):
            x1, y1, x2, y2 = bbx[j]

            #####Handle zero-height/width due to rounding errors -- inflate bb by 1 or 2 pixel####
            if (x1 == x2):
                x1 -= 1
                x2 += 1
            if (y1 == y2):
                y1 -= 1
                y2 += 1

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            ##############################################################################

            try:
                if avail[j]:
                    part = img[y1:y2, x1:x2]  # crop a part from rgb image
                    part = cv2.resize(part, (64, 64))  # resize to part-mask dimensions
                    part[mmx[j] == 0] = np.array([0, 0, 0])  # keep only required part of rgb using mask
                else:
                    part = np.zeros((64, 64, 3))  # black part-rgb if some part is not there in rgb
            except:
                part = np.zeros(
                    (64, 64, 3))  # Handle exceptional cases of negaitive values in bb: make a black part rgb
                mask[idx, j, :, :, 0] = np.zeros((64, 64))  # Black part mask
                box[idx, j] = np.zeros(5)  # delete gibberish bounding box-- zero out

                print(
                    'object {} at index {} skipped due to error in bounding box dimensions :({},{}),({},{})....'.format(
                        object_name, idx, x1, x2, y1, y2))

            parts.append(part)

        """   #uncomment to change bb values to a fraction of rgb image instead of fraction of canvas size
        box[idx,:,1:]*=canvas_size
        box[idx,:,1:]/=np.array([h,w,h,w])
        """  # uncomment to change bb values to a fraction of rgb image instead of fraction of canvas size

        parts = np.array(parts).astype('uint8')  # for memory issues change to uint8
        cropped.append(parts)

    cropped = np.array(cropped)

    return box, mask.astype('bool'), cropped, images


def visualise(num_samples):
    idx = np.random.randint(0, class_v.shape[0], num_samples)
    for i in idx:
        img = rgbs[i]
        for j in range(24):
            if X_train[i][j][0] == 1:
                h, w = img.shape[:2]
                box = np.round(X_train[i, j, 1:] * canvas_size).astype('uint16')
                st = tuple(box[:2])
                en = tuple(box[2:])
                img1 = img.copy()
                cv2.rectangle(img1, pt1=st, pt2=en, color=(255, 0, 0), thickness=2)
                plt.imshow(img1)
                plt.show()
                plt.imshow(masks[i, j, :, :, 0])
                plt.show()
                plt.imshow(rgb_masks[i, j])
                plt.show()


def make_data(object_class_names, train_mode):
    class_v = {}
    X_train = {}
    nX_train = {}
    adj_train = {}
    masks = {}
    rgb_masks = {}
    rgbs = {}
    train_split = 0.75
    valid_split = 0.10
    test_split = 0.15
    for object_name in object_class_names:
        path = 'data/'
        print(type(object_class_names))
        if object_name == 'motorbike' or object_name == 'bicycle':
            path = 'motor&cycle/'
        with open(path + object_name + '_images', 'rb') as f:
            o_images = pickle.load(f)
        with open(path + object_name + '_part_separated_labels', 'rb') as f:
            o_labels = pickle.load(f)
        with open(path + object_name + '_part_separated_masks', 'rb') as f:
            o_masks = pickle.load(f)
        with open(path + object_name + '_part_separated_bbx', 'rb') as f:
            o_bbx = pickle.load(f)

        
      

        samples = o_labels.shape[0]
        print("samples = " + str(samples))
        train_set_limit = int(train_split * samples)
        valid_set_limit = int(valid_split * samples)
        test_set_limit = int(test_split * samples)

        if train_mode=='train':
          label = o_labels[0:train_set_limit]
          box = o_bbx[0:train_set_limit]
          mask = o_masks[0:train_set_limit]
          image = o_images[0:train_set_limit]
        if train_mode=='valid':
          label = o_labels[train_set_limit:train_set_limit + valid_set_limit]
          box = o_bbx[train_set_limit:train_set_limit + valid_set_limit]
          mask = o_masks[train_set_limit:train_set_limit + valid_set_limit]
          image = o_images[train_set_limit:train_set_limit + valid_set_limit]
        if train_mode=='test':
          label = o_labels[train_set_limit + valid_set_limit:]
          box = o_bbx[train_set_limit + valid_set_limit:]
          mask = o_masks[train_set_limit + valid_set_limit:]
          image = o_images[train_set_limit + valid_set_limit:]

        if train_mode=='test':
          sample_size = len(box)
          new_labels = []
          new_box = []
          new_mask = []
          new_image = []

          for i in range(100):
            if i >= sample_size:
              idx = i%sample_size
            else:
              idx = i
            new_labels.append(label[idx])
            new_box.append(box[idx])
            new_mask.append(mask[idx])
            new_image.append(image[idx])

          label = new_labels
          box = new_box
          mask = new_mask
          image = new_image

        print("Masks= " + str(len(label)))
        print("Images= " + str(len(image)))

        max_parts = len(label[0])

        flipped_label = []
        flipped_box = []
        flipped_mask = []
        flipped_image = []

        angle = 3
        if train_mode=='train':
          for l, b, m, i in zip(label, box, mask, image):
              ll, bb, mm, ii = flip_data_instance(l, b, m, i, object_name)
              flipped_label.append(ll)
              flipped_box.append(bb)
              flipped_mask.append(mm)
              flipped_image.append(ii)

          flipped_label = np.asarray(flipped_label)
          flipped_box = np.asarray(flipped_box)
          flipped_mask = np.asarray(flipped_mask)
          flipped_image = np.asarray(flipped_image)



          label = np.concatenate((label, flipped_label), axis=0)

          box = np.concatenate((box, flipped_box), axis=0)
          mask = np.concatenate((mask, flipped_mask), axis=0)
          image = np.concatenate((image, flipped_image), axis=0)

        box = append_labels(box)
   
        box = pad_along_axis(box, 24, axis=1)
        mask = make_full_mask_list(mask, 24)
   
        numparts = 24
        final_input = []
        for i, x in enumerate(box):
            l = numparts
            temp = np.zeros((l, l), dtype=np.float32)
            for j in range(l):
                temp[j][j] = 1
                if x[j][0] == 1:
                    for y in tree[object_name][j]:
                        if x[y][0] == 1:
                            temp[j][y] = 1
                            temp[y][j] = 1
            final_input.append(temp)
        adj = np.asarray(final_input)

        box, mask, rgb_mask, image = get_rgbmask(box, mask, image, object_name)
        class_v[object_name] = (np.asarray([np.eye(10)[class_dic[object_name]]] * len(box))).astype('bool')
        X_train[object_name] = box
        adj_train[object_name] = (adj).astype('bool')
        masks[object_name] = mask.astype('bool')
        rgb_masks[object_name] = rgb_mask.astype('uint8')
        rgbs[object_name] = image


    class_v = np.concatenate(list(class_v.values()), axis=0)
    X_train = np.concatenate(list(X_train.values()), axis=0)
    rgb_imgs = np.concatenate(list(rgbs.values()), axis=0)
    masks = np.concatenate(list(masks.values()), axis=0)
    adj_train = np.concatenate(list(adj_train.values()), axis=0)
    rgb_masks = np.concatenate(list(rgb_masks.values()), axis=0)


    print("Part Masks Size = :" + str(masks.shape))
    print("BBox Size = :" + str(X_train.shape))
    print("RGB Part Size = :" + str(rgb_masks.shape))
    print("RGB Shape = :" + str(rgb_imgs.shape))
    print("Classes Shape " + str(class_v.shape))
    print(class_v[0:5])

    if train_mode=='train':
      rand_indices = np.random.permutation(masks.shape[0])
      X_train = X_train[rand_indices]
      masks = masks[rand_indices]
      rgb_masks = rgb_masks[rand_indices]
      rgb_imgs = rgb_imgs[rand_indices]
      class_v = class_v[rand_indices]
      

    return X_train, masks, rgb_masks, rgb_imgs, class_v.astype(dtype=np.float32)
    # visualise(5)


def data_prep(X_train, masks, rgb_masks, classes, model='seq2seq'):

    indices = np.where(X_train[:, :, 0] == 1)
    if model == 'partae':
        part_ae_masks = masks[indices]
        part_ae_rgb_masks = rgb_masks[indices]
        classes = np.expand_dims(classes, axis=1)
        classes = np.tile(classes, (1, 24, 1))
        classes = classes[indices]

        return part_ae_masks, part_ae_rgb_masks, classes.astype(dtype=np.float32)

    else:
        part_number = np.zeros(shape=(masks.shape[0], 24, 24))
        for i in range(X_train.shape[0]):
            cnt = 0
            for j in range(X_train.shape[1]):
                if X_train[i][j][0] == 1:
                    part_number[i][j][j] = 1
        sign = np.zeros(shape=(masks.shape[0], masks.shape[1], 1))
        for i in range(X_train.shape[0]):
            stop = 0
            for j in range(X_train.shape[1]):
                if X_train[i][j][0] == 1:
                    stop = j

            sign[i][stop][0] = 1

        bce_mask = X_train[:, :, 0]
        affine_input = X_train[:, :, 1:]
        affine_target = X_train[:, :, 1:]

 
        return rgb_masks, affine_input, part_number, affine_target, sign, bce_mask
