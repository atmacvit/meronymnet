import streamlit as st
import pandas as pd
import numpy as np
import os, cv2
import sys
import scipy.io
import random
from pathlib import Path  
import json
import PIL
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

def main():
    # Render the readme as markdown using st.markdown.
    
    f = open(os.path.join(path.parent.parent,'experiment_scripts','visualization','instructions.md'), "r")
    readme_text = st.markdown(f.read())
    
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        f = open(source+'src/visualization/'+"visualization.py", "r")
        st.code(f.read())
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()
        
def run_the_app():
    
    labels = os.listdir(anno_source)
    object_type = st.sidebar.selectbox("Search for which objects?", labels, 1)
    
    image_list = os.listdir(os.path.join(anno_source,object_type,'bbox'))
    image_list = [name.split('.')[0] for name in image_list]

    image_list = np.append(image_list,['Random'])
    selected_image = st.sidebar.selectbox("Select image", image_list, 1)
    
    n_labels = len(labels)
    diff = int(360/(n_labels+2))
    colours = np.arange(0,360,diff)
    colours = np.uint8([[[col,255,255] for col in colours[1:-1]]])
    colours = cv2.cvtColor(colours,cv2.COLOR_HSV2RGB)[0]
    label_col = colours[np.where(np.array(labels)==object_type)][0]
    
    part_disp = st.checkbox("Display parts")
    bbox_disp = st.checkbox("Display bounding box view")
    obbox_disp = st.checkbox("Display oriented bounding box view")
    
    display_multi(object_type,image_list,selected_image,label_col,part_disp,bbox_disp,obbox_disp)
    
#     if part_disp:
#         display_parts(object_type,image_list,selected_image,label_col)
#     else:
#         display_objects(object_type,image_list,selected_image,label_col)

def display_objects(object_type,image_list,selected_image,label_col,):
    
    disp_image = []
    
    if selected_image=='Random':
        disp_list = image_list[:-1]
        random.shuffle(disp_list)
        disp_list = disp_list[:4]

    else:
        disp_list=[selected_image]
        
    for image in disp_list:
        with open(os.path.join(anno_source,object_type,'bbox',image+'.json')) as fp:
            annotation_dict = json.load(fp)
        temp_image = cv2.imread(img_source+image+'.jpg')
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
        for idx in annotation_dict.keys():
            contours, _ = cv2.findContours(np.uint8(np.matrix(annotation_dict[idx]['mask'])),
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            temp_image = cv2.drawContours(temp_image, contours, -1,label_col.tolist() , 1)
        disp_image.append(temp_image)
     
    st.subheader('Ground Truth')
    
    for idx,img in enumerate(disp_image):
        st.image(img,caption=disp_list[idx])

def display_multi(object_type,image_list,selected_image,label_col,part_disp,bbox_disp,obbox_disp):
    
    font                   = cv2.FONT_HERSHEY_PLAIN
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    
#     if part_disp & bbox_disp:
#         n=3
#     elif part_disp|bbox_disp:
#         n=2
#     else:
#         n=1
    n = 1+sum([part_disp,bbox_disp,obbox_disp])
    
    disp_image = []
    if selected_image=='Random':
        disp_list = image_list[:-1]
        random.shuffle(disp_list)
        disp_list = disp_list[:4]
    else:
        disp_list=[selected_image]
    
    for image in disp_list:
        with open(os.path.join(anno_source,object_type,'bbox',image+'.json')) as fp:
            annotation_dict = json.load(fp)
        raw_image = cv2.imread(os.path.join(img_source,image+'.jpg'))
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1,n, figsize = (12,4))
        ob_image = raw_image.copy()
        part_image = raw_image.copy()
        obb_image = raw_image.copy()
        t=0
        im_x_min,im_y_min,im_x_max,im_y_max = raw_image.shape[0],raw_image.shape[1],0,0
        for idx in annotation_dict.keys():
            ob_x_min,ob_y_min,ob_x_max,ob_y_max =annotation_dict[idx]['bbox']
            im_x_min = min(im_x_min,ob_x_min)
            im_y_min = min(im_y_min,ob_y_min)
            im_x_max = max(im_x_max,ob_x_max)
            im_y_max = max(im_y_max,ob_y_max)
            
        for idx in annotation_dict.keys():
            contours, _ = cv2.findContours(np.uint8(np.matrix(annotation_dict[idx]['mask'])),
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            ob_image = cv2.drawContours(ob_image, contours, -1,label_col.tolist() , 2)
            if n>1:
                axes[0].imshow(ob_image)
                plt.axis('off')  
            else:
                axes.imshow(ob_image)
                plt.axis('off') 
            
#             ob_x_min,ob_y_min,ob_x_max,ob_y_max =annotation_dict[idx]['bbox']
#             im_x_min = min(im_x_min,ob_x_min)
#             im_y_min = min(im_y_min,ob_y_min)
#             im_x_max = max(im_x_max,ob_x_max)
#             im_y_max = max(im_y_max,ob_y_max)
            part_dict = annotation_dict[idx]['parts']
            if part_dict and part_disp:
                for part in part_dict.keys():
                    contours, _ = cv2.findContours(np.uint8(np.matrix(part_dict[part]['mask'])),
                               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    part_image = cv2.drawContours(part_image, contours, -1,label_col.tolist() , 1)
                    loc = np.mean(contours[0],axis=0).astype(int)[0].tolist()
                    part_image = cv2.putText(part_image,part[0],(loc[0],loc[1]),font,fontScale,fontColor,lineType)
            
            if part_dict and obbox_disp:
                for part in part_dict.keys():
                    contours,_ = cv2.findContours(np.uint8(np.matrix(part_dict[part]['mask'])),
                               cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
                    
                    for c in contours:
                        rect = cv2.minAreaRect(c)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        obb_image = cv2.drawContours(obb_image, [box],0,(0,0,255),1)
                        
            if part_dict and bbox_disp:
                bb_image = raw_image.copy()
                if part_disp:
                    axes[2].imshow(raw_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1])
                    plt.axis('off')  
                    t = 2
                else:
                    axes[1].imshow(raw_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1])
                    plt.axis('off')  
                    t = 1
                for part in part_dict.keys():
                    
                    x_min,y_min,x_max,y_max =part_dict[part]['bbox']
                    rect = patches.Rectangle((y_min-im_y_min,x_min-im_x_min),y_max-y_min,x_max-x_min,
                                             linewidth=1,edgecolor='r',facecolor='none')
            
                    axes[t].add_patch(rect)
            if part_disp:
                axes[1].imshow(part_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1])
            if obbox_disp:
                axes[t+1].imshow(obb_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1])
            
        st.pyplot(fig)
        #disp_image.append(temp_image)
     
    st.subheader('Ground Truth')
    
    
    #for idx,img in enumerate(disp_image):
    #    st.image(img,caption=disp_list[idx])

def display_parts(object_type,image_list,selected_image,label_col):
    
    font                   = cv2.FONT_HERSHEY_PLAIN
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 1

    disp_image = []
    if selected_image=='Random':
        disp_list = image_list[:-1]
        random.shuffle(disp_list)
        disp_list = disp_list[:4]
    else:
        disp_list=[selected_image]
    
    for image in disp_list:
        with open(os.path.join(anno_source,object_type,'bbox',image+'.json')) as fp:
            annotation_dict = json.load(fp)
        temp_image = cv2.imread(os.path.join(img_source,image+'.jpg'))
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
        
        for idx in annotation_dict.keys():
            contours, _ = cv2.findContours(np.uint8(np.matrix(annotation_dict[idx]['mask'])),
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            temp_image = cv2.drawContours(temp_image, contours, -1,label_col.tolist() , 1)
        
            part_dict = annotation_dict[idx]['parts']
            if part_dict:
                for part in part_dict.keys():
                    contours, _ = cv2.findContours(np.uint8(np.matrix(part_dict[part]['mask'])),
                               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    temp_image = cv2.drawContours(temp_image, contours, -1,label_col.tolist() , 1)
                    loc = np.mean(contours[0],axis=0).astype(int)[0].tolist()
                    temp_image = cv2.putText(temp_image,part[0][0],(loc[0],loc[1]),font,fontScale,fontColor,lineType)
                        
        disp_image.append(temp_image)
     
    st.subheader('Ground Truth')
    
    
    for idx,img in enumerate(disp_image):
        st.image(img,caption=disp_list[idx])
        

if __name__ == "__main__":
    
    curr_path = os.getcwd()
    path = Path(curr_path)
    anno_source = os.path.join(path.parent.parent,'dataset_files','datasets','PASCAL-VOC','xybb-objects')
    img_source = os.path.join(path.parent.parent,'dataset_files','datasets','PASCAL-VOC','scene')
    main()
