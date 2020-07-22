from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from scores import fid,mis,ins,ds,ams,cas,utils
import argparse

class Metrics:

  def __init__(self,path_fake,path_real=None,model="inception",path_model=None,splits=10,object_names=None,num_samples=500):

    self.path_fake=path_fake #Generated data
    self.path_real=path_real #Training data

    #If object names is not given then infer them based on folders in the generations directory
    if object_names is None: self.object_names=utils.Utils.get_object_names(self.path_fake)
    else: self.object_names=object_names

    #Number of splits per-class to average the scores
    self.splits=int(splits)
    
    #Number of samples to draw from each class
    self.num_samples=int(num_samples)


    #Select required model, its prerocessing function and expected input shape
    if model=='inception':
      self.model = InceptionV3()
      self.input_shape=(299,299,3)
      self.preprocess=keras.applications.inception_v3.preprocess_input
    elif model=='resnet50':
      self.model=keras.models.load_model(path_model)
      self.input_shape=(224,224,3)
      self.preprocess=keras.applications.resnet50.preprocess_input
    else:
      raise NameError(model+' not implemented') 
    
    #Dictionary of available quantitative measures
    self.score_functions={
      'FID':self.Fid,
      'AMS':self.Ams,
      'IS':self.Is,
      'MIS':self.Mis,
      'DS':self.Ds,
      'CAS':self.Cas,
    }

  
  def Fid(self): 

    #FID is calculated only when real dataset is provided
    if self.path_real is not None:
      #remove top layer of model for FID calculation
      model_notop= keras.Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
      return fid.FID(self.path_real,self.path_fake,model_notop,self.preprocess,self.input_shape,self.splits,self.object_names,self.num_samples).calculate()

  def Ams(self):

    #AMS is calculated only when real data set is provided
    if self.path_real is not None:
      return ams.AMS(self.path_real,self.path_fake,self.model,self.preprocess,self.input_shape,self.splits,self.object_names,self.num_samples).calculate()

  def Is(self):

    return ins.IS(self.path_fake,self.model,self.preprocess,self.input_shape,self.splits,self.object_names,self.num_samples).calculate()

  def Mis(self):

    return mis.MIS(self.path_fake,self.model,self.preprocess,self.input_shape,self.splits,self.object_names,self.num_samples).calculate()

  def Ds(self):

    return ds.DS(self.path_fake,self.model,self.preprocess,self.input_shape,self.splits,self.object_names,self.num_samples).calculate()

  def Cas(self):
    #CAS is calculated only when real dataset is provided
    if self.path_real is not None:
      #remove top layer of model for CAS calculation
      model_notop= keras.Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
      return cas.CAS(self.path_real,self.path_fake,model_notop,self.preprocess,self.input_shape,self.splits,self.object_names,self.num_samples).calculate()


if __name__ == '__main__': 

  """
  path_fake[string]="path/to/generations_folder"
  path_real [string] (optional)="path/to/training_data" (if not provided then FID and AMS are not calculated)
  model [string] (optional) {'inception','resnet50'}=whether to use pretrained inceptionV3 or fine tuned resnet50 model (deafult='inception')
  path_model [string] (optional)="path/to/saved_model" [only used when inception is not true]
  splits [int] (optional)=Number for splits per-class (default=10)
  precision [int] (optional)=Number of decimal digits accurate to which results are required (default=4)
  object_names [string] (optional)=comma separated string without any space like "cow,person,cat" (defaults to all object names)
  scores [string] (optional)=comma separated string of score abbreviations without any space like "FID,MIS,IS,AMS,DS,CAS" (defaults to all scores)
  path_result [string] (optional)="path/to/result.csv" [if not provided then a suitable path is inferred]
  num_samples [int] (optional) = Number of samples of each class for which results are calculated (default = 500)

  sample-use:
  python 'path/to/metrics.py' "$path_fake" --path_real "$path_real" --splits="$splits" --model "$model" --object_names "$object_names" --path_model "$path_model" --path_result "$path_result" --precision="$precision" --num_samples="$num_samples"

  Saves a csv file of metrics in results folder.
  """

  #Parsing commandline arguments
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('path_fake', type=str, metavar='path_fake',help='diectory of generations')

  parser.add_argument('--path_real', type=str, default=None,help='directory of ground truth data')

  parser.add_argument('--path_model', type=str, default=None,help='path of model')
    
  parser.add_argument('--splits', type=int, default=10,help='Number of splits to average the score over')

  parser.add_argument('--model', type=str, default="inception", choices=['inception','resnet50'],help='whether to use pretrained inceptionV3 or fine tuned resnet50 for score calculation')

  parser.add_argument('--object_names', type=str, default=None, help="Comma separated string of objects without any space (Ex: 'cat,cow,dog')-- If none then all objects are considered")

  parser.add_argument('--scores', type=str, default='fid,mis,is,ams,ds,cas', help="Comma separated string of scores abbreviations without any space (Ex: 'FID,MIS,IS,AMS,DS,CAS')")

  parser.add_argument('--path_result', type=str, default=None, help='path of result with .csv extension')

  parser.add_argument('--precision', type=int, default=4, help='required precision while calculating mean and std')

  parser.add_argument('--num_samples', type=int, default=500, help='Number of samples of each class for which results are calculated')


  args = parser.parse_args()
  path_fake = args.path_fake
  path_real = args.path_real
  path_model = args.path_model
  path_result = args.path_result
  splits = args.splits
  model = args.model
  object_names = args.object_names
  precision = args.precision
  num_samples = args.num_samples
  score_names = args.scores.split(',')
  if object_names is not None: object_names=object_names.split(',')


  #Create an instance of Metrics class
  metric=Metrics(path_fake,path_real,model,path_model,splits,object_names,num_samples) 

  #Calculate the required scores
  scores={}
  for score_name in score_names:
    scores[score_name.upper()]=metric.score_functions[score_name.upper()]()
  
  #Make a dataframe from scores
  df=utils.Utils.make_dataframe(scores,precision)

  #Find path(if not given by user) & make a directory for results if not present already
  path_result=utils.Utils.find_path(path_result,path_fake,model)

  #Save formatted and unformatted results
  utils.Utils.save_results(df,scores,path_result)   
  
  
