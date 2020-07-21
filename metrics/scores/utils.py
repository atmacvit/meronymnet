import glob
import cv2
import numpy as np
import pandas as pd
import os

class Utils:

  @staticmethod
  def find_path(path_result,path_fake,inception):
    #if path_result is not given by user then infer a suitable path based on generations name and type of model
    if path_result is None:
      separated=path_fake.split(os.path.sep)
      separated[-2]='results'
      if inception.lower()=='true': extension='_inception.csv'
      else: extension='_resnet.csv'
      path_result=(os.path.sep).join(separated)+extension
    
    #create results directory if not present already
    dir_result=os.path.dirname(path_result)
    if not os.path.isdir(dir_result):
      os.makedirs(dir_result)
    
    return path_result

  #load images in a directory and resize them
  @staticmethod
  def load_images(image_dir,input_shape):

    images = []
    files=glob.glob(image_dir)

    for filepath in files:

        image=cv2.imread(filepath)
        image=image[:,:,::-1]
        image=cv2.resize(image,input_shape[:2])
        images.append(image)
      
    images=np.array(images)
    return images

  #Obtain image sub-directory for a particular object
  @staticmethod
  def get_path(path,obj):
    if path[-1]!=os.path.sep:
      return path+os.path.sep+obj+os.path.sep+'*'
    else:
      return path+obj+os.path.sep+'*'

  #covert a tuple (mean,std) to mean ± std for all values in dictionary
  @staticmethod
  def to_string(score,precision):
    string_score={}
    for obj in score:
      if obj!='mean':
        string_score[obj]=str(round(score[obj][0],precision))+' ± '+str(round(score[obj][1],precision))
      else:
        string_score[obj]=round(score[obj],precision)
    return string_score

  #Make a combined dataframe from dictionaries containing all types of scores
  @staticmethod
  def make_dataframe(scores):
    dfs={}
    for score in scores:
      string_scores=Utils.to_string(scores[score])
      dfs[score]=pd.DataFrame(string_scores.items(), columns=['Object', score])
      dfs[score]=dfs[score].set_index('Object',drop=True)

    df=pd.concat(list(dfs.values()),axis=1)
    return df