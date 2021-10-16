from skimage.color import rgb2gray
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from skimage.io import imsave, imread

mypath1 = './datasets/visw/'
mypath2 = './datasets/RGB/'
onlyfiles = sorted(([f for f in listdir(mypath1) if isfile(join(mypath1, f))]))
# print(onlyfiles)
for i in onlyfiles:
    # img2 = imread('./gen/gen/'+i)
    # try:
    img1 = cv2.imread('./datasets/visw/'+i)
    # print(img1)
    img2 = cv2.imread('./datasets/RGB/'+i)
    # print(img2)
    # img1 = img1[:,:,::-1]
    # img2 = img2[:,:,::-1]
	# print(img2)
    background= np.sum(img1,axis=-1)==0
    # print(background) 
    background= np.stack([background,background,background],axis=-1)
    # print(img2)
    img2[background]= 255 #Change background to zero

    img44 = img2
    # img2 = crop_zoom(img2)
    # print(img2)
    # i = i.replace('horse', 'aeroplane')
    cv2.imwrite('../RGBs/'+ i, img2)
    # except:
    #     cv2.imwrite('./white/'+ i, img44)
    #     continue