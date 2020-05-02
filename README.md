# Code for the VMF-HMM
---

### File
datas  # container of data  
results # container figure of syn dataset and experimental results  
config # the hyper parameters of dataset  
model # model code  
utils # some util function  
train # training code  

---
### Requirements
matplotlib  
numpy  
pandas  
scipy  
sklearn  
nilearn  

---
### Run code
__params:__  
-data_name dataset name  
-verbose print information or not  

-N the number of state  
-K the number of mixture  
-converge the over threshold of runing  
-max_iter max iterations of training  

__example:__  
python train.py -data_name small_data -verbose 1 -N 2 -K 2 -converge 1+1e-4 -max_iter 100
