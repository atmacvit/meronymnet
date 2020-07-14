import numpy as np
from scipy.linalg import sqrtm
from .utils import Utils

class FID:

  def __init__(self,path_real,path_fake,model,preprocess,input_shape,splits,object_names):

    self.path_real=path_real
    self.path_fake=path_fake
    self.model=model
    self.preprocess=preprocess
    self.input_shape=input_shape
    self.object_names=object_names
    self.splits= splits

  #find mean and std for FID of one class
  def calculate_fid(self,activation1, activation2):

    fids=[]

    num_img=activation1.shape[0]

    for i in range(self.splits):

      act1=activation1[i*num_img//self.splits:(i+1)*num_img//self.splits]
      act2=activation2[i*num_img//self.splits:(i+1)*num_img//self.splits]

      # calculate mean and covariance statistics
      mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
      mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
      # calculate sum squared difference between means
      ssdiff = np.sum((mu1 - mu2)**2.0)
      # calculate sqrt of product between cov
      covmean = sqrtm(sigma1.dot(sigma2))
      # check and correct imaginary numbers from sqrt
      if np.iscomplexobj(covmean):
        covmean = covmean.real
      # calculate score
      fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

      fids.append(fid)


    return np.mean(fids), np.std(fids)**2

  def calculate(self):

    fids={}

    for obj in self.object_names:

      #load and preprocess data
      obj_path_real=Utils.get_path(self.path_real,obj)
      obj_path_fake=Utils.get_path(self.path_fake,obj)
      im_real=self.preprocess(Utils.load_images(obj_path_real,self.input_shape))
      im_fake=self.preprocess(Utils.load_images(obj_path_fake,self.input_shape))

      #get predictions
      act1 = self.model.predict(im_real)
      act2 = self.model.predict(im_fake)

      #calculate scores
      fid=self.calculate_fid(act1,act2)
      fids[obj]=fid

    #calculate mean over all classes
    fids['mean']=np.mean(list(map((lambda x: x[0]), list(fids.values()))))

    return fids