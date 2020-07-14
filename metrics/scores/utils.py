import glob
import cv2
import numpy as np
import pandas as pd
import os

class Utils:

  #load images in a directory and resize them
  @staticmethod
  def load_images(image_dir,input_shape):

    images = []

    for filepath in glob.glob(image_dir):

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

  #covert a tuple (mean,variance) to mean ± variance for all values in dictionary
  @staticmethod
  def to_string(score):
    string_score={}
    for obj in score:
      if obj!='mean':
        string_score[obj]=str(round(score[obj][0],2))+' ± '+str(round(score[obj][1],2))
      else:
        string_score[obj]=round(score[obj],2)
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