# MoDriver
MoDriver is a coding and non-coding cancer driver discovery tool based on imbalanced deep learning. The framework's input is multi-omics profiles (such as mutation, CNA, mRNA) of all the tumor samples. The output is the corresponding driver score, p-value, q-value for each locus. MoDriver is mainly divided into two components: 1. Multiple input-single output deep network is used to generate the original driving element score. 2. A module for generating simulated mutations corresponding to each cancer set and determining the p-values and q-values of each locus.
```{r}
# the input data file is features of multi-omics data and runs the following command to finish training  processes: 
python MoDriver.py -m train -t Pancan
# Train the model on the Pan-cancer dataset 
```
MoDriver's scoring module is used as follows: 
```{r}
python MoDriver.py -m score -t Pancan
# Generate the cancer driver scores on the Pan-cancer dataset 
```
To support MoDriver on multiple cancer types, we use the python script run_all.py to automate the scoring process:
```{r}
python run_all.py -m score
``` 
MoDriver's performance comparison module is used as follows: 
```{r} 
python run_all.py -m compare
```  
MoDriver is based on Python. The imbalanced deep learning network's implementation was based on the open-source library Numpy 1.19.1, scikit-learn 0.23.2, Keras 2.3.1, and Tensorflow 1.14.0 (GPU version). After testing, this framework has been working correctly on Ubuntu Linux release 20.04. We used the NVIDIA Tesla T4 (16G) for  model training and testing.
