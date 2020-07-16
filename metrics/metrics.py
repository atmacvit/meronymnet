from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from scores import fid,mis,ins,ds,ams,utils
import argparse
import os

class Metrics:

  def __init__(self,path_fake,path_real=None,inception=True,path_model=None,splits=10,object_names=None):

    self.path_fake=path_fake #Generated data
    self.path_real=path_real #Training data

    if object_names is None:
      self.object_names=['cow', 'person', 'cat', 'dog', 'horse', 'sheep', 'bird', 'aeroplane', 'motorbike','bicycle']
    else:
      self.object_names=object_names

    #Number of splits per-class to average the scores
    self.splits=int(splits)

    #If inception is set to false, use any other pretrained resnet50 model and its preprocessing function
    if inception.lower()=='true':
      self.input_shape=(299,299,3)
      self.model = InceptionV3()
      self.preprocess=keras.applications.inception_v3.preprocess_input
    else:
      self.model=keras.models.load_model(path_model)
      self.input_shape=(224,224,3)
      self.preprocess=keras.applications.resnet50.preprocess_input

  
  def Fid(self):

    #remove top layer of model for FID calculation
    model_notop= keras.Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
    return fid.FID(self.path_real,self.path_fake,model_notop,self.preprocess,self.input_shape,self.splits,self.object_names).calculate()

  def Ams(self):

    return ams.AMS(self.path_real,self.path_fake,self.model,self.preprocess,self.input_shape,self.splits,self.object_names).calculate()

  def Is(self):

    return ins.IS(self.path_fake,self.model,self.preprocess,self.input_shape,self.splits,self.object_names).calculate()

  def Mis(self):

    return mis.MIS(self.path_fake,self.model,self.preprocess,self.input_shape,self.splits,self.object_names).calculate()

  def Ds(self):

    return ds.DS(self.path_fake,self.model,self.preprocess,self.input_shape,self.splits,self.object_names).calculate()

if __name__ == '__main__': 

  """
  path_fake[string]="path/to/generations_folder"
  path_real [string] (optional)="path/to/training_data" (if not provided then FID and AMS are not calculated)
  inception [string] (optional) {'True','False'}=whether to use inceptionV3 model (deafult='True')
  path_model [string] (optional)="path/to/saved_model" [only used when inception is not true]
  splits [int] (optional)=Number for splits per-class (default=10)
  object_names [string] (optional)=comma separated string without any space like "cow,person,cat" (defaults to all object names)
  path_result [string] (optional)="path/to/result.csv" [if not provided then a suitable path is inferred]

  sample-use:
  python 'path/to/metrics.py' "$path_fake" --path_real "$path_real" --splits="$splits" --inception "$inception" --object_names "$object_names" --path_model "$path_model" --path_result "$path_result"

  Saves a csv file of metrics in results folder.
  """

  #parsing commandline arguments
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('path_fake', type=str, metavar='path_fake',help='diectory of generations')

  parser.add_argument('--path_real', type=str, default=None,help='directory of ground truth data')

  parser.add_argument('--path_model', type=str, default=None,help='path of model')
    
  parser.add_argument('--splits', type=int, default=10,help='Number of splits to average the score over')

  parser.add_argument('--inception', type=str, default="True", choices=['true','True','False','false'],help='whether to use inceptionV3 for score calculation')

  parser.add_argument('--object_names', type=str, default=None, help="Comma separated string of objects without any space (Ex: 'cat,cow,dog')-- If none then all objects are considered")

  parser.add_argument('--path_result', type=str, default=None, help='path of result with .csv extension')


  args = parser.parse_args()
    
  path_fake = args.path_fake
  path_real = args.path_real
  path_model = args.path_model
  path_result = args.path_result
  splits = args.splits
  inception = args.inception
  object_names = args.object_names

  if object_names is not None:
    object_names=object_names.split(',')

  metric=Metrics(path_fake,path_real,inception,path_model,splits,object_names) 

  scores={}
  scores['IS']=metric.Is()
  scores['MIS']=metric.Mis()
  scores['DS']=metric.Ds()

  #calculate FID and AMS only if ground truth data is also available
  if path_real is not None: 
    scores['FID']=metric.Fid()
    scores['AMS']=metric.Ams()

  #make a dataframe from scores
  df=utils.Utils.make_dataframe(scores)

  #if path_result is not given then infer a suitable path based on generations name and type of model
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

  #save results   
  df.to_csv(path_result)
  
