# fMRI TimeSeries Classification using Transformer Encoder
The aim of this project is to investigate the use of Transformer networks for classifying and representing brain responses to movies as measured by fMRI scans. Transformers, which have become a central method in natural language processing, are characterized by their ability to find correlations between different words in sentences or temporal sequences and emphasize important parts of the data using the Attention mechanism. In this project, we propose using a Transformer network to analyze temporal signals obtained from fMRI scans, aiming to answer the following questions: Can a Transformer network classify brain data obtained from different movies? What can be learned from the representations of the Transformer's attention head for data originating from different brain regions? 
 
In this project, the Value matrice generated by multi head attention has been used for classification. A classification mechanism has been added to the end of the encoder part of the model. 
 

You can find hyper parameters in model_summary.py file. 
The output for model summary is represented below. 

![image](https://user-images.githubusercontent.com/6734818/225657838-b3b211b1-9412-4752-ab98-059051f61060.png)


TRAINING:  
  
DATASET:   
We load the data from a local directory which contains pickle files ordered by ROI and subject number. each pickle file represent the fmri brain activity of a single subject during all movies.
We prepare the data for Train/Test using the preprocess2.py file which convert all relevant pickles into Dataloaders.
There are 14 different classes and the all classes are enumerated from 1 to 14 in label dataset.    
seq_len= Durtion of the movie # fmri scan duraion during the movie watching   
feature= Number of voxels # depends on the ROI we're training the model on
Input size for training : [batch_size, seq_len, feature].     


TEST RESULTS: 
you can see and compare target labels and predicted classes.  
![image](https://user-images.githubusercontent.com/6734818/226144528-31dea983-508c-4ee7-818f-c7a29607f242.png)       




