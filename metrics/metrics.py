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
  inception [bool] (optional)=whether to use inceptionV3 model (deafult=True)
  path_model [string] (optional)="path/to/saved_model" [only used when inception is not true]
  splits [int] (optional)=Number for splits per-class (default=10)
  object_names [string] (optional)=comma separated string without any space like "cow,person,cat" (defaults to all object names)

  sample-use:
  python 'path/to/metrics.py' --set path_fake="$path_fake" path_real="$path_real" splits="$splits" inception="$inception" object_names="$object_names" path_model="$path_model"

  Saves a csv file of metrics in results folder. Results folder must be there in the parent of parent folder of generations. Name of csv file is same as name of generations folder.
  """

  #parsing commandline arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--set",
                        metavar="KEY=VALUE",
                        nargs='+',
                        help="Set a number of key-value pairs "
                             "(do not put spaces before or after the = sign). "
                             "If a value contains spaces, you should define "
                             "it with double quotes: "
                             'foo="this is a sentence". Note that '
                             "values are always treated as strings.")
  args = parser.parse_args()
  arguments = utils.Utils.parse_vars(args.set)

  #Handling optional/default arguments
  if 'path_real' not in arguments:
    arguments['path_real']=None

  if 'path_model' not in arguments:
    arguments['path_model']=None

  if 'inception' not in arguments:
    arguments['inception']='True'

  if 'splits' not in arguments:
    arguments['splits']=10

  if 'object_names' not in arguments:
    arguments['object_names']=None
  else:
    arguments['object_names']=arguments['object_names'].split(',')

  metric=Metrics(**arguments) 

  scores={}
  scores['IS']=metric.Is()
  scores['MIS']=metric.Mis()
  scores['DS']=metric.Ds()

  #calculate FID and AMS only if training data is also available
  if arguments['path_real'] is not None: 
    scores['FID']=metric.Fid()
    scores['AMS']=metric.Ams()

  #make a dataframe from scores
  df=utils.Utils.make_dataframe(scores)

  #save results in results folder with same name as generations folder
  separated=arguments['path_fake'].split(os.path.sep)
  separated[-2]='results'
  result_path=(os.path.sep).join(separated)+'.csv'
  df.to_csv(result_path)

