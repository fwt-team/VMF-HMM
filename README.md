# Code for the 'Sequentially spherical data modeling with hidden Markov models and its application to fMRI data analysis'
---

### File
datas  # container of data  
results # experimental results  
config # the hyper parameters of experiments  
model # model code  
utils # some util function  
train # training code  

---
### Requirements
matplotlib==3.2.1  
numpy==1.18.2  
pandas==1.0.3  
scipy==1.4.1  
sklearn==0.22.2  
nilearn==0.6.2  

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

---
### Reference
If you use our code in your work, please cite our paper. 

    @article{FAN2020106341,
    title = "Sequentially spherical data modeling with hidden Markov models and its application to fMRI data analysis",
    author = "Wentao Fan and Lin Yang and Nizar Bouguila and Yewang Chen",
    journal = "Knowledge-Based Systems",
    volume = "206",
    pages = "106341",
    year = "2020"
    }
