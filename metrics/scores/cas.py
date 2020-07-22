from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from .utils import Utils


class CAS:

  def __init__(self,path_real,path_fake,model,preprocess,input_shape,splits,object_names,num_samples):

    self.path_real=path_real
    self.path_fake=path_fake
    self.model=model
    self.preprocess=preprocess
    self.input_shape=input_shape
    self.object_names=object_names
    self.splits= splits
    self.num_samples=num_samples

  #find mean and std for FID of one class
  def calculate_cas(self,knn_model,preds_fake,labels_fake):

    cass=[]
    num_img=preds_fake.shape[0]

    for i in range(self.splits):

      preds_fake_split=preds_fake[i*num_img//self.splits:(i+1)*num_img//self.splits]
      labels_fake_split=labels_fake[i*num_img//self.splits:(i+1)*num_img//self.splits]

      cas=knn_model.score(preds_fake_split,labels_fake_split)
      cass.append(cas)

    return np.mean(cass), np.std(cass)

  #fit KNN model
  def fit_knn(self):

    preds_real=[]
    labels_real=[]

    for i,obj in enumerate(self.object_names):

      #load and preprocess data
      obj_path_real=Utils.get_path(self.path_real,obj)
      im_real=self.preprocess(Utils.load_images(obj_path_real,self.input_shape,self.num_samples))

      #get predictions and labels
      preds_real.append(self.model.predict(im_real))
      labels_real.append(np.ones(im_real.shape[0])*i)
    
    #concatenate preds and labels into numpy array
    preds_real=np.concatenate(preds_real,axis=0)
    labels_real=np.concatenate(labels_real,axis=0)

    knn_model = KNeighborsClassifier(n_neighbors=20,weights='distance')
    knn_model.fit(preds_real,labels_real)
    
    return knn_model

  def calculate(self):

    #Fit a knn model on real data
    knn_model=self.fit_knn()

    cass={}

    for i,obj in enumerate(self.object_names):

      #load and preprocess fake data
      obj_path_fake=Utils.get_path(self.path_fake,obj)
      im_fake=self.preprocess(Utils.load_images(obj_path_fake,self.input_shape,self.num_samples))

      #get predictions and labels
      preds_fake=self.model.predict(im_fake)
      labels_fake=np.ones(preds_fake.shape[0])*i

      #calculate cas 
      cas=self.calculate_cas(knn_model,preds_fake,labels_fake)
      cass[obj]=cas

    #calculate mean over all classes
    cass['mean']=np.mean(list(map((lambda x: x[0]), list(cass.values()))))

    return cass