import tensorflow as tf
import numpy as np
from scipy.stats import entropy
from .utils import Utils

class DS:

  def __init__(self,path,model,preprocess,input_shape,splits,object_names,num_samples):

    self.path=path
    self.model=model
    self.preprocess=preprocess
    self.input_shape=input_shape
    self.object_names=object_names
    self.splits= splits
    self.num_samples=num_samples

  #calculate intra class diversity based on ms-ssim
  def calc_intra(self,images):

    intra=0

    for i,image1 in enumerate(images):
      for j,image2 in enumerate(images):

        ms_ssim=tf.image.ssim_multiscale(
                                  image1, image2, max_val=255
                              )
        intra+=ms_ssim
          
    intra=1-(intra/((images.shape[0])**2))
    return intra

  #calculate inter class diversity
  def calc_inter(self,img):

    #preprocess and predict
    img=self.preprocess(img)
    preds=self.model.predict(img)
    num_class=preds.shape[-1]

    argmax=np.argmax(preds,axis=-1)
    cx= np.eye(num_class)[argmax] #one-hot prediction vector

    avg_cx=np.sum(cx,axis=0)/preds.shape[0]
    inter=entropy(avg_cx)/(np.log(num_class)) #entropy of avg one-hot prediction vector

    return inter

  #find mean and std for diversity score of one class
  def calc_ds(self,images):

    dss=[]
    num_img=images.shape[0]

    for i in range(self.splits):
      
      img=images[i*num_img//self.splits:(i+1)*num_img//self.splits]
      dintra=self.calc_intra(img)
      dinter=self.calc_inter(img)
      ds=(dinter*dintra)**(0.5)
      dss.append(ds)

    return np.mean(dss), np.std(dss)

  def calculate(self):

    ds={}

    for obj in self.object_names:

      #load data
      obj_path=Utils.get_path(self.path,obj)
      images=Utils.load_images(obj_path,self.input_shape,self.num_samples)

      #calculate scores
      score=self.calc_ds(images)
      ds[obj]=score
      
    #calculate mean over all classes
    ds['mean']=np.mean(list(map((lambda x: x[0]), list(ds.values()))))
    
    return ds
    