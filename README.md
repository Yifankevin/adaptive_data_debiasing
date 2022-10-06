# Adaptive Data Debiasing through Bounded Exploration

By [Yifan Yang](https://sites.google.com/view/yangyifan/yifan_yang)<sup>1</sup>, [Yang Liu](http://www.yliuu.com/)<sup>2</sup> and [Parinaz Naghizadeh](https://parinazn.com/)<sup>1</sup>  

  <sup>1</sup>: The Ohio State University, Columbus OH, USA  
  <sup>2</sup>: University of California, Santa Cruz, Santa Cruz CA, USA 

### Introduction

This repository is for our NeurIPS 2022 paper '[Adaptive Data Debiasing through Bounded Exploration](https://arxiv.org/abs/2110.13054)'.

### Required packages

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

### Baseline models  

Mode1 1: Exploititaiton only model (Collect all data above threshold)  
- See file  

Model 2: Pure exploration (Collect all data without considering risk)  
- See file



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
