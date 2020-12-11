# Metrics for quantitative evaluation

To measure the performance of our generative model, we have implemented 6 commonly used metrics:
1. [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500)
2. [Activation Maximization Score](https://arxiv.org/abs/1703.02000)


## How to use?

Package scores contains implementation of above mentioned scores apart from a module for utility functions.

The script metrics.py contains a class Metrics which can be imported and used to calculate the required scores. 

#### Use as a script

```
$ python 'path/to/metrics.py' "$path_fake" --path_real "$path_real" --splits="$splits" --model "$model" --object_names "$object_names" --path_model "$path_model" --path_result "$path_result" --precision="$precision" --num_samples="$num_samples"
```
Saves a csv file of metrics and a JSON file of raw results in results folder.

```
$ python 'path/to/metrics.py' --help 
```
Shows help message for the script.

##### Positional Arguments:
```
path_fake [string] ="path/to/generations_folder"
```
##### Optional Arguments:
```
path_real [string] = "path/to/training_data" (if not provided then FID and AMS are not calculated)

model [string] {'inception','resnet50'} = whether to use pretrained inceptionV3 or fine tuned resnet50 model (default='inception')

path_model [string] = "path/to/saved_model" [only used when model is not 'inception']

splits [int] = Number for splits per-class (default=10)

precision [int] = Number of decimal digits accurate to which results are calculated (default=4)

object_names [string] = comma separated string without any space like "cow,person,cat" (defaults to all object in generations directory)

scores [string] = comma separated string of score abbreviations without any space like "FID,AMS" (defaults to all scores)

path_result [string] = "path/to/result_file.csv" (path must have .csv extennsion, if path is not provided then a suitable path is inferred)

num_samples [int] = Number of samples of each class for which results are calculated (default = 500)
```

#### Use as module
```
from metrics import Metrics

metric= Metrics(path_fake,path_real,model,path_model,splits,object_names,num_samples) #Note: Here object_names should a list of objects and NOT a string

scores={'FID':None,  'AMS':None} #scores to calculate

for score_name in scores:
	scores[score_name]=metric.score_functions[score_name]()
	
#Each value in scores dictionary contains a dictionary of a particular type of score with keys as object names and values as a tuple of mean and std
print(scores['FID']) 
```
## Description of Scores
Throughout this description, we have followed the following convention for notations for the sake of consistency (unless otherwise specified).
* <img src="https://render.githubusercontent.com/render/math?math=X">= Real data distribution
* <img src="https://render.githubusercontent.com/render/math?math=G">= Generated data distribution or equivalently the generator function
* <img src="https://render.githubusercontent.com/render/math?math=(x,y)">= Real sample and its label <img src="https://render.githubusercontent.com/render/math?math=(x~X)">
* <img src="https://render.githubusercontent.com/render/math?math=(x_g,y_g)">= Generated sample and its label <img src="https://render.githubusercontent.com/render/math?math=(x_g~G)">
* <img src="https://render.githubusercontent.com/render/math?math=z">= Latent vector
* <img src="https://render.githubusercontent.com/render/math?math=N">= Number of classes


### Fréchet Inception Distance

FID quantifies the quality of generated samples by first embedding into a feature space given by (a specific layer) of Inception Net. Then, viewing the embedding layer as a continuous multivariate Gaussian, the mean and covariance is estimated for both the generated data and the real data. The Fréchet distance between these two Gaussians is then used to quantify the quality of the samples, i.e. 

![equation](https://latex.codecogs.com/gif.latex?FID%28X%2CG%29%20%3D%20%7B%5CVert%20%5Cmu_X%20-%20%5Cmu_G%20%5CVert%7D_2%5E2&plus;%20Tr%28%5CSigma_G%20&plus;%20%5CSigma_X%20-%202%28%5CSigma_G%5CSigma_X%29%5E%7B%5Cfrac12%7D%29)
 
where <img src="https://render.githubusercontent.com/render/math?math=(\mu_X,\Sigma_X)"> and <img src="https://render.githubusercontent.com/render/math?math=(\mu_G,\Sigma_G)"> are the mean and covariance of the sample embeddings from the data distribution and model distribution respectively. 

The authors show that the score is consistent with human judgment and more robust to noise than IS . Unlike IS, FID can detect intra-class mode dropping, i.e. a model that generates only one image per class can score a perfect IS, but will have a bad FID. A significant drawback of both IS and FID measures is the inability to detect overfitting. A “memory GAN” which stores all training samples would score perfectly. 


### Activation Maximization Score

The entropy term on <img src="https://render.githubusercontent.com/render/math?math=y_g"> in Inception Score is problematic when training data is not evenly distributed over classes, for that <img src="https://render.githubusercontent.com/render/math?math=\argmin H(y_g)"> is a uniform distribution. To take into account the class imbalance in training set, AM score replaces the <img src="https://render.githubusercontent.com/render/math?math=\ H(y_g)"> term in Inception score with the KL divergence between <img src="https://render.githubusercontent.com/render/math?math=\ p(y_g)"> and  <img src="https://render.githubusercontent.com/render/math?math=\ p(y)"> . The AM score is then defined as:

![equation](https://latex.codecogs.com/gif.latex?AMS%28X%2CG%29%3D%20%5Cmathbb%7BKL%7D%28p%28y%29%20%7C%7C%20p%28y_g%29%29%20&plus;%20E_%7Bx_g%7D%5BH%28y_g%7Cx_g%29%5D)

The AM score consists of two terms. The first one is minimized when <img src="https://render.githubusercontent.com/render/math?math=\ p(y_g)"> is close to <img src="https://render.githubusercontent.com/render/math?math=\ p(y)">. The second term is minimized when the predicted class label for generated samples has low entropy. The minimal value of AM Score is zero, and the smaller value, the better.



## References
* [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
* [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
* [DeLiGAN : Generative Adversarial Networks for Diverse and Limited Data](https://arxiv.org/abs/1706.02071)
* [ACTIVATION MAXIMIZATION GENERATIVE ADVERSARIAL NETS](https://arxiv.org/abs/1703.02000)
* [Multi-Modal Generative Adversarial Networks for Diverse Datasets](https://openreview.net/forum?id=rkgWBi09Ym)
* [Pros and Cons of GAN Evaluation Measures](https://arxiv.org/abs/1802.03446)
* [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/abs/1711.10337)
