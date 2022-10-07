# Adaptive Data Debiasing through Bounded Exploration

By [Yifan Yang](https://sites.google.com/view/yangyifan/yifan_yang)<sup>1</sup>, [Yang Liu](http://www.yliuu.com/)<sup>2</sup> and [Parinaz Naghizadeh](https://parinazn.com/)<sup>1</sup>  

  <sup>1</sup>: The Ohio State University, Columbus OH, USA  
  <sup>2</sup>: University of California, Santa Cruz, Santa Cruz CA, USA 

### Introduction

This repository is for our NeurIPS 2022 paper '[Adaptive Data Debiasing through Bounded Exploration](https://arxiv.org/abs/2110.13054)'.

### Required packages/modules

from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.utils import shuffle  
from sklearn import metrics  
from sklearn.metrics import confusion_matrix  
import sklearn.preprocessing as preprocessing  

from scipy.stats import norm, beta  
from statistics import median  
from random import choices  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

from responsibly.dataset import build_FICO_dataset  
from responsibly.fairness.interventions.threshold import find_thresholds  
 

### Assumptions  

1: Individuals have single dimensional feature or score $x \in \mathcal{R}$  
2: Individuals belong to one of two groups: $G_a, G_b$  
3: Individuals are either qualified or unqualified, and have a true label $y \in$  {0,1}  
4: A threshold-based classifier $\theta_g$ is used to make 'accept/reject' decision on these individuals  
5: Biased estimated feature distribution $\hat{f}^y_g(x)$ and true distribution $f^y_g(x)$ are differ in single parameter  
6: Measure of bias: differences between two parameters $|\hat{\omega}^y_g - \omega^y_g|$

### Experiments : baseline models  

Model 1: Exploititaiton only model (Collect all data above threshold)  
- See file [Exploitation_Only_Baseline.ipynb](Exploitation_Only_Baseline.ipynb) 

Model 2: Pure exploration (Collect all data without considering risk)  
- See file [Pure_Exploration_Baseline.ipynb](Pure_Exploration_Baseline.ipynb)

### Experiments: symmetric and asymmetric distributions

Synthetic symmetric Data: see file [Gaussian_Experiment.ipynb](Gaussian_Experiment.ipynb)  
Synthetic asymmetric Data: see file [Beta_Experiment.ipynb](Beta_Experiment.ipynb)  

### Experiments: real dataset and fairness

Real Dataset (**Adult, FICO, Fair**): see file [Adult_FICO_Gaussian_Fair.ipynb](Adult_FICO_Gaussian_Fair.ipynb)  


### Citation

If this repository is useful for your research, please consider citing:  

    @article{yang2022adaptive,  
    title={Adaptive Data Debiasing through Bounded Exploration},  
    author={Yifan Yang, Yang Liu and Parinaz Naghizadeh},  
    journal={Advances in Neural Information Processing Systems}, 
    volume={35},
    year={2022}. 
    }
   

### Acknowledgement  
We sincerely thank three reviewers for taking the time to review our paper and providing constructive feedback to improve our paper. We are also grateful for support from Cisco Research, and the NSF program on Fairness in AI in collaboration with Amazon under Award No. IIS-2040800. Any opinion, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF, Amazon, or Cisco.
