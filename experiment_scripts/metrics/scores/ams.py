import numpy as np
from .utils import Utils

class AMS:

  def __init__(self,path_real,path_fake,model,preprocess,input_shape,splits,object_names,num_samples):

    self.path_real=path_real
    self.path_fake=path_fake
    self.model=model
    self.preprocess=preprocess
    self.input_shape=input_shape
    self.object_names=object_names
    self.splits= splits
    self.num_samples=num_samples

  
  #calculate am_score for one split
  def am_score(self,preds, ref_preds):

    preds = preds + 1e-18
    am_per = np.mean(-np.sum(preds * np.log(preds), 1)) #Entropy term

    avg_preds = np.mean(preds, 0)
    ref_avg_preds = np.mean(ref_preds, 0)
    am_avg = -np.sum(ref_avg_preds * np.log(avg_preds / ref_avg_preds), 0) #KL-div term

    score = am_per + am_avg

    return score

  #find mean and std for am_score of one class
  def calculate_am_score(self,preds, ref_preds):
    scores = []
    for i in range(self.splits):
        part = preds[(i * preds.shape[0] // self.splits):((i + 1) * preds.shape[0] // self.splits), :]
        scores.append(self.am_score(part, ref_preds))
    return np.mean(scores), np.std(scores)

  def calculate(self):

    am_scores={}
    sz = []
    for obj in self.object_names:
      
      #load and preprocess data
      obj_path_real=Utils.get_path(self.path_real,obj)
      obj_path_fake=Utils.get_path(self.path_fake,obj)
      im_real=self.preprocess(Utils.load_images(obj_path_real,self.input_shape,self.num_samples))
      im_fake=self.preprocess(Utils.load_images(obj_path_fake,self.input_shape,self.num_samples))

      #get predictions
      preds_real= self.model.predict(im_real)
      preds_fake= self.model.predict(im_fake)

      #calculate scores
      score=self.calculate_am_score(preds_fake, preds_real)
      am_scores[obj]=(score)
      sz.append(self.num_samples)

    #calculate mean over all classes
    #am_scores['mean']=np.mean(list(map((lambda x: x[0]), list(am_scores.values()))))
    means = list(map((lambda x: x[0]), list(am_scores.values())))
    stds = list(map((lambda x: x[1]), list(am_scores.values())))
    am_scores['aggregate'] = Utils.obtain_aggregate_stats(means, stds, sz)
    sorted_means = sorted(means)
    am_scores['median'] = (sorted_means[4] + sorted_means[5])/2.0
    

    
    return am_scores
