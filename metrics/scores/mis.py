import numpy as np
from .utils import Utils

class MIS:

  def __init__(self,path,model,preprocess,input_shape,splits,object_names,num_samples):

    self.path=path
    self.model=model
    self.preprocess=preprocess
    self.input_shape=input_shape
    self.object_names=object_names
    self.splits= splits
    self.num_samples=num_samples

  #find mean and std for MIS of one class
  def calculate_mis(self,part):

    logp= np.log(part)
    self_ = np.sum(part*logp,axis=1)
    cross = np.mean(np.dot(part,np.transpose(logp)),axis=1)
    diff = self_ - cross
    kl1 = []

    for j in range(self.splits):
        diffj = diff[(j * diff.shape[0] // self.splits):((j+ 1) * diff.shape[0] //self.splits)]
        kl1.append(np.exp(diffj.mean()))

    mis_mean=np.mean(kl1)
    mis_std=np.std(kl1)
    
    return mis_mean, mis_std

  def calculate(self):

    mis={}
    sz = []
    for obj in self.object_names:

      #load and preprocess data
      obj_path=Utils.get_path(self.path,obj)
      images=self.preprocess(Utils.load_images(obj_path,self.input_shape,self.num_samples))
      
      #find predictions
      preds = self.model.predict(images)

      #calculate scores
      score=self.calculate_mis(preds)
      mis[obj]=score
      sz.append(self.num_samples)

    #calculate mean over all classes
    means = list(map((lambda x: x[0]), list(mis.values())))
    stds = list(map((lambda x: x[1]), list(mis.values())))
    #mis['mean'] = mean_agg
    mis['aggregate'] = Utils.obtain_aggregate_stats(means, stds, sz)
    #mis['std'] = std_agg
    
    
    return mis
    