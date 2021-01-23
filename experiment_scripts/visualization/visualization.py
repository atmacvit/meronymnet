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

def main():
    # Render the readme as markdown using st.markdown.
    
    f = open(os.path.join(path.parent.parent,'experiment_scripts/visualization/instructions.md'), "r")
    readme_text = st.markdown(f.read())
    
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        f = open(source+'src/visualization/'+"temp.py", "r")
        st.code(f.read())
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()
        
def run_the_app():
    
    labels = os.listdir(anno_source)
    object_type = st.sidebar.selectbox("Search for which objects?", labels, 1)
    
    image_list = os.listdir(anno_source+object_type+'\\bbox')
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
    
    if part_disp:
        display_parts(object_type,image_list,selected_image,label_col)
    else:
        display_objects(object_type,image_list,selected_image,label_col)

def display_objects(object_type,image_list,selected_image,label_col,):
    
    disp_image = []
    
    if selected_image=='Random':
        disp_list = image_list[:-1]
        random.shuffle(disp_list)
        disp_list = disp_list[:4]

    else:
        disp_list=[selected_image]
        
    for image in disp_list:
        with open(anno_source+object_type+'\\bbox\\'+image+'.json') as fp:
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

def display_multi(object_type,image_list,selected_image,label_col,part_dist,bbox_disp):
    
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
        with open(anno_source+object_type+'\\bbox\\'+image+'.json') as fp:
            annotation_dict = json.load(fp)
        temp_image = cv2.imread(img_source+image+'.jpg')
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
        with open(anno_source+object_type+'\\bbox\\'+image+'.json') as fp:
            annotation_dict = json.load(fp)
        temp_image = cv2.imread(img_source+image+'.jpg')
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
    anno_source = os.path.join(path.parent.parent,'dataset_files\\datasets\\PASCAL-VOC\\xybb-objects\\')
    img_source = os.path.join(path.parent.parent,'dataset_files\\datasets\\PASCAL-VOC\\scene\\')
    #source = 'C:/Users/user/Documents/Workspace/New folder/cvpr-mero/'
    #anno_source = 'C:/Users/user/Documents/Workspace/New folder/cvpr-mero/images/Annotations_Part/'
    main()
