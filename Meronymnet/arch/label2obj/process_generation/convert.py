from skimage.color import rgb2gray
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from skimage.io import imsave, imread

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

def crop_zoom(img1):

    background= np.sum(img1,axis=-1)==(255*3) #All white background
    background= np.stack([background,background,background],axis=-1)
    img1[background]=0 #Change background to zero
    xmin, ymin, xmax, ymax = cordinates(rgb2gray(img1))
    img1=img1[ymin:ymax, xmin:xmax][...,::-1]
    # plt.imshow(img1)
    # plt.show()
    height,width = (-ymin+ymax), (-xmin+xmax)
    psize = height
    if width>height:
        psize=width
    img1 = cv2.copyMakeBorder(img1,(psize-height)//2,(psize-height)//2,(psize-width)//2,(psize-width)//2,cv2.BORDER_CONSTANT,value=[0.,0.,0.])
    try:
        img1=cv2.resize(img1,(128,128),interpolation=cv2.INTER_AREA)
    except:
        img1 = img1
    return img1

mypath = 'maskResults/'
onlyfiles = sorted(([f for f in listdir(mypath) if isfile(join(mypath, f))]))
print(onlyfiles)
for i in onlyfiles:
    # img2 = imread('./gen/gen/'+i)
    try:
        img2 = cv2.imread('maskResults/'+i)
        # print(img2)
        img2 = img2[:,:,::-1]
    	# print(img2)
        img44 = img2
        img2 = crop_zoom(img2)
        # print(img2)
        # i = i.replace('horse', 'aeroplane')
        cv2.imwrite('C-SPADE/datasets/visw/'+ i, img2)
    except:
        cv2.imwrite('C-SPADE/datasets/visw/'+ i, img44)
        continue
