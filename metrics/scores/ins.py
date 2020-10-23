import numpy as np
from .utils import Utils

class IS:

  def __init__(self,path,model,preprocess,input_shape,splits,object_names,num_samples):

    self.path=path
    self.model=model
    self.preprocess=preprocess
    self.input_shape=input_shape
    self.object_names=object_names
    self.splits= splits
    self.num_samples=num_samples

  #find mean and std for IS of one class
  def calculate_is(self,images):

    is_scores=[]

    num_img=images.shape[0]

    for i in range(self.splits):

      img=images[i*num_img//self.splits:(i+1)*num_img//self.splits]
      yhat = self.model.predict(img)
      eps=1e-16
      p_yx = yhat
      # calculate p(y)
      p_y = np.expand_dims(p_yx.mean(axis=0), 0)
      # calculate KL divergence using log probabilities
      kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
      # sum over classes
      sum_kl_d = kl_d.sum(axis=1)
      # average over images
      avg_kl_d = np.mean(sum_kl_d)
      # undo the log
      is_score = np.exp(avg_kl_d)

      is_scores.append(is_score)

    return np.mean(is_score), np.std(is_scores)

  def calculate(self):

    incp_scores={}
    sz = []

    for obj in self.object_names:

      #load and preprocess data
      obj_path=Utils.get_path(self.path,obj)
      images=self.preprocess(Utils.load_images(obj_path,self.input_shape,self.num_samples))
      
      #calculate scores
      insc=self.calculate_is(images)
      incp_scores[obj]=insc
      sz.append(self.num_samples)

    #calculate mean over all classes
    #incp_scores['mean']=np.mean(list(map((lambda x: x[0]), list(incp_scores.values()))))
    means = list(map((lambda x: x[0]), list(incp_scores.values())))
    stds = list(map((lambda x: x[1]), list(incp_scores.values())))
    incp_scores['aggregate'] = Utils.obtain_aggregate_stats(means, stds, sz)
    return incp_scores