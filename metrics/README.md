# Metrics for quantitative evaluation

To measure the performance of our generative model, we have implemented 6 commonly used metrics:
1. [Inception Score](https://arxiv.org/abs/1606.03498)
1. [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500)
1. [Modified Inception Score](https://arxiv.org/abs/1706.02071)
1. [Activation Maximization Score](https://arxiv.org/abs/1703.02000)
1. [Diversity Score](https://openreview.net/forum?id=rkgWBi09Ym)
1. [Classification Accuracy Score](https://arxiv.org/abs/1905.10887)

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
path_real [string] = "path/to/training_data" (if not provided then FID, CAS and AMS are not calculated)

model [string] {'inception','resnet50'} = whether to use pretrained inceptionV3 or fine tuned resnet50 model (default='inception')

path_model [string] = "path/to/saved_model" [only used when model is not 'inception']

splits [int] = Number for splits per-class (default=10)

precision [int] = Number of decimal digits accurate to which results are calculated (default=4)

object_names [string] = comma separated string without any space like "cow,person,cat" (defaults to all object in generations directory)

scores [string] = comma separated string of score abbreviations without any space like "FID,MIS,IS,AMS,DS,CAS" (defaults to all scores)

path_result [string] = "path/to/result_file.csv" (path must have .csv extennsion, if path is not provided then a suitable path is inferred)

num_samples [int] = Number of samples of each class for which results are calculated (default = 500)
```

#### Use as module
```
from metrics import Metrics

metric= Metrics(path_fake,path_real,model,path_model,splits,object_names,num_samples) #Note: Here object_names should a list of objects and NOT a string

scores={'FID':None, 'MIS':None, 'IS':None, 'AMS':None, 'CAS':None, 'DS':None} #scores to calculate

for score_name in scores:
	scores[score_name]=metric.score_functions[score_name]()
	
#Each value in scores dictionary contains a dictionary of a particular type of score with keys as object names and values as a tuple of mean and std
print(scores['IS']) 
```
## Description of Scores
Throughout this description, we have followed the following convention for notations for the sake of consistency (unless otherwise specified).
* <img src="https://render.githubusercontent.com/render/math?math=X">= Real data distribution
* <img src="https://render.githubusercontent.com/render/math?math=G">= Generated data distribution or equivalently the generator function
* <img src="https://render.githubusercontent.com/render/math?math=(x,y)">= Real sample and its label <img src="https://render.githubusercontent.com/render/math?math=(x~X)">
* <img src="https://render.githubusercontent.com/render/math?math=(x_g,y_g)">= Generated sample and its label <img src="https://render.githubusercontent.com/render/math?math=(x_g~G)">
* <img src="https://render.githubusercontent.com/render/math?math=z">= Latent vector
* <img src="https://render.githubusercontent.com/render/math?math=N">= Number of classes

### Inception Score

Inception Score offers a way to quantitatively evaluate the quality of generated samples. The score was motivated by the following considerations: 

1. The conditional label distribution of samples containing meaningful objects should have low entropy
2.  The variability of the samples should be high, or equivalently, the marginal <img src="https://render.githubusercontent.com/render/math?math=\int p(y_g|x_g = G(z))dz"> should have high entropy.

Thus, these two probability distributions should be very different from each other and the KL Divergence between them should be high. Hence, these are combined into one score as:

![equation](https://latex.codecogs.com/gif.latex?IS%28G%29%20%3D%20%7B%5Crm%20e%7D%5E%7BE_%7Bx_g%7D%5B%5Cmathbb%7BKL%7D%28p%28y_g%20%7C%20x_g%29%7C%7C%20p%28y_g%29%29%5D%7D%20%3D%20%7B%5Crm%20e%7D%5E%7B%20H%28y_g%29%20-%20E_%7Bx_g%7D%20%5BH%28y_g%7Cx_g%29%5D%20%7D)

Exponentiation is performed so that results are easier to compare.
The classifier is Inception Net (version 3) trained on Image Net. The authors found that this score is well-correlated with scores from human annotators. Drawbacks include insensitivity to the prior distribution over labels and not being a proper distance.

### Fréchet Inception Distance

FID quantifies the quality of generated samples by first embedding into a feature space given by (a specific layer) of Inception Net. Then, viewing the embedding layer as a continuous multivariate Gaussian, the mean and covariance is estimated for both the generated data and the real data. The Fréchet distance between these two Gaussians is then used to quantify the quality of the samples, i.e. 

![equation](https://latex.codecogs.com/gif.latex?FID%28X%2CG%29%20%3D%20%7B%5CVert%20%5Cmu_X%20-%20%5Cmu_G%20%5CVert%7D_2%5E2&plus;%20Tr%28%5CSigma_G%20&plus;%20%5CSigma_X%20-%202%28%5CSigma_G%5CSigma_X%29%5E%7B%5Cfrac12%7D%29)
 
where <img src="https://render.githubusercontent.com/render/math?math=(\mu_X,\Sigma_X)"> and <img src="https://render.githubusercontent.com/render/math?math=(\mu_G,\Sigma_G)"> are the mean and covariance of the sample embeddings from the data distribution and model distribution respectively. 

The authors show that the score is consistent with human judgment and more robust to noise than IS . Unlike IS, FID can detect intra-class mode dropping, i.e. a model that generates only one image per class can score a perfect IS, but will have a bad FID. A significant drawback of both IS and FID measures is the inability to detect overfitting. A “memory GAN” which stores all training samples would score perfectly. 

### Modified Inception Score

In its original formulation, Inception Score assigns a higher score for models that result in a low entropy class conditional distribution <img src="https://render.githubusercontent.com/render/math?math=p(y_g|x_g)">. However, it is desirable to have diversity within image samples of a particular category. To characterise this diversity, MIS uses a cross-entropy style score <img src="https://render.githubusercontent.com/render/math?math=-p(y_g|x_{g_i}) \log(p(y_g|x_{g_j})"> where <img src="https://render.githubusercontent.com/render/math?math=x_j">   is sample of the same class as <img src="https://render.githubusercontent.com/render/math?math=x_i">  as per the outputs of the trained inception model. Incorporating this term into the original inception-score results in: 

![equation](https://latex.codecogs.com/gif.latex?MIS%28G%29%3D%20%7B%5Crm%20e%7D%5E%7BE_%7Bx_%7Bg_i%7D%7D%5BE_%7Bx_%7Bg_j%7D%7D%5B%5Cmathbb%7BKL%7D%28p%28y%7Cx_%7Bg_i%7D%29%7C%7Cp%28y%7Cx_%7Bg_j%7D%29%29%5D%5D%7D)

which is calculated on a per-class basis and is then averaged over all classes. 

Essentially, MIS can be viewed as a proxy for measuring both intra-class sample diversity as well as sample quality.

### Activation Maximization Score

The entropy term on <img src="https://render.githubusercontent.com/render/math?math=y_g"> in Inception Score is problematic when training data is not evenly distributed over classes, for that <img src="https://render.githubusercontent.com/render/math?math=\argmin H(y_g)"> is a uniform distribution. To take into account the class imbalance in training set, AM score replaces the <img src="https://render.githubusercontent.com/render/math?math=\ H(y_g)"> term in Inception score with the KL divergence between <img src="https://render.githubusercontent.com/render/math?math=\ p(y_g)"> and  <img src="https://render.githubusercontent.com/render/math?math=\ p(y)"> . The AM score is then defined as:

![equation](https://latex.codecogs.com/gif.latex?AMS%28X%2CG%29%3D%20%5Cmathbb%7BKL%7D%28p%28y%29%20%7C%7C%20p%28y_g%29%29%20&plus;%20E_%7Bx_g%7D%5BH%28y_g%7Cx_g%29%5D)

The AM score consists of two terms. The first one is minimized when <img src="https://render.githubusercontent.com/render/math?math=\ p(y_g)"> is close to <img src="https://render.githubusercontent.com/render/math?math=\ p(y)">. The second term is minimized when the predicted class label for generated samples has low entropy. The minimal value of AM Score is zero, and the smaller value, the better.

### Diversity Score

To measure the diversity of generated samples, DS takes into account both the inter-class, and the intra-class diversity. 

Intra-class diversity is measured by the average (negative) MS-SSIM metric between all pairs of generated images in a given set of generated images:

![equation](https://latex.codecogs.com/gif.latex?d_%7Bintra%7D%28G%29%20%3D%201-%20%5Cfrac1%7B%7CG%7C%5E2%7D%20%5Csum_%7B%28x_g%2Cx_g%27%29%5Cin%20G%5Ctimes%20G%7D%7BMS%20%7B%5Ctext%20-%7D%20SSIM%28x_g%2Cx_g%27%29%7D)

where G is a set of generated samples.

For inter-class diversity, a pre-trained classifier is used to classify the set of generated images, such that for each sampled image <img src="https://render.githubusercontent.com/render/math?math=\ x_g">, there is a classification prediction in the form of a one-hot vector <img src="https://render.githubusercontent.com/render/math?math=\ c(x_g)">.  Then, the entropy of the average one-hot classification prediction vector is measured to evaluate the diversity between classes in the samples set: 

![equation](https://latex.codecogs.com/gif.latex?d_%7Binter%7D%28G%29%3D%20%5Cfrac1%7B%5Clog%28N%29%7DH%20%5Cleft%28%5Cfrac1%7B%7CG%7C%7D%20%5Csum_%7Bx_g%20%5Cin%20G%7D%20c%28x_g%29%5Cright%29)

Finally, the diversity score is defined as the geometric mean of intra class diversity and inter-class diversity

![equation](https://latex.codecogs.com/gif.latex?d%28G%29%3D%20%5Csqrt%7Bd_%7Bintra%7D%28G%29%20%5Cast%20d_%7Binter%7D%28G%29%20%7D)

### Classification Accuracy Score

CAS  is based on the idea that if the model captures the data distribution, performance on any downstream task should be similar whether using the original or model data.  Accordingly, a discriminative model (classifier) is trained on the synthetic data <img src="https://render.githubusercontent.com/render/math?math=\ (x_g,y_g)"> ,  and the performance of the classifier is evaluated on the real data. The accuracy thus obtained is called the Classification Accuracy Score (CAS). The authors further note that simply having a good CAS  score does not imply that the generative model accurately modelled the data distribution. This may occur due to a variety of reason. As an example, a generative model that memorizes the training set will achieve a high CAS score.

## References
* [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
* [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
* [DeLiGAN : Generative Adversarial Networks for Diverse and Limited Data](https://arxiv.org/abs/1706.02071)
* [ACTIVATION MAXIMIZATION GENERATIVE ADVERSARIAL NETS](https://arxiv.org/abs/1703.02000)
* [Multi-Modal Generative Adversarial Networks for Diverse Datasets](https://openreview.net/forum?id=rkgWBi09Ym)
* [Classification Accuracy Score for Conditional Generative Models](https://arxiv.org/abs/1905.10887)
* [Pros and Cons of GAN Evaluation Measures](https://arxiv.org/abs/1802.03446)
* [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/abs/1711.10337)
