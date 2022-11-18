# Prediction of missing peaks in GCMS spectra
## Content 
- source code of a [Bachelor thesis](https://is.muni.cz/th/dz62r/?lang=en) research project  
- source code of evaluation of the research project on the real-world GCMS data measured at RECETOX centre.
## Description
Compound identification is essential for monitoring the environment. Gas Chromatography-Mass Spectrometry (GC-MS) is a widely used method for such identification. A crucial step in the processing of complex data coming from the physical GC-MS instrument is peak detection. The errors of peak detection algorithms, such as missed peaks, severely limit the researchers' ability to monitor low-concentration toxins and pollutants. We proposed and developed an approach to predict the missing peaks with the Machine Learning (ML) methods leveraging the spectral databases. We experimented with multiple ML models based on kNN and neural networks. We observed that the attention-based transformer model is the most suitable for the prediction of peaks missed due to noise. Furthermore, the multi-layer perceptron turned out to be superior in predicting peaks missed because of imperfect chromatographic separation. These models have been quantitatively and qualitatively shown to predict multiple correct peaks for both known and unknown compounds. Moreover, we identified which kinds of peaks are less and more challenging to predict. The results illustrate the feasibility of ML methods trained on spectral databases to improve peak detection and establish the ground for future research

## How to reproduce 
### Part 1 - dataset exploration, ML models' training, evaluation on syntetic data
This work uses NIST EI database data to predict the missing peaks. 
It should be runnable on any low-resolution data mass spectral data (no guarantees). 

#### Setting up the envinroment
1. \[Recommended\] Contact us, so we can supply the customized Singularity container, which can be run in any custom Singularity/Docker environment
2. \[Alternative\] Install the packages from requirements.txt (should work with Python 3.8.5 but may need some relaxation of packages versions) 

#### Setting up the data
1. Download the database in .msp format
2. Open the preprocessing/data_splitting.ipynb notebook and customize the paths (and code if needed)
3. Split the data by running the customized notebook 
4. Run the exploration/explore_data.ipynb notebook to understand the data more

#### Severity estimation
1. Open and run the exploration/matching_model.ipynb notebook to calculate the matches
2. Open and run the exploration/explore_severity.ipynb notebook to plot the recalls

#### Gas2Vec training
1. Open and run Gas2Vec.ipynb to train the gas2vec model in scenario A
2. Select the best version and save (rename) it to "gas2vec/in_database.model"

#### Missing peaks models
1. Open and run the knn_model_SpeckNN.ipynb to compute/visualise the predictions of Spectral kNN for both problems (may take several hours/days depending on the dataset size)
2. Open and run the knn_model_Gas2VeckNN.ipynb to compute/visualise the predictions of Gas2Vec kNN
3. Open and run the generative_model_LSTM.ipynb to train and compute/visualise the predictions of LSTM models (A100 or other GPU recommended)
4. Open and run the generative_model_Decoder.ipynb to train and compute/visualise the predictions of Decoder model with selectd parameters (A100 GPU recommended)
5. \[Optional\] Experiment with training other Decoder versions by changing the congfig parameters in the generative_model_Decoder.ipynb notebook 
6. Open and run the feedforward_model_LR_MLP.ipynb to train and compute/visualize the predictions of linear model and MLP models

#### Missing peaks models variants comparison
1. Open and customize exploration/explore_training.ipynb with selected model for which the training progress should be plotted
2. Open and run/customize exploration/explore_evaluation.ipynb to plot the comparison of models' variants on the validation set, select the best variant of each model

#### Evaluation of missing peaks models
1. Open, customize, and run the evaluation/evaluationA.ipynb to compare the best variants of each model quantitavely 
2. Open, customize, and run the exploration/explore_visual.ipynb to visualize the predictions of best models qualitatevely in coloured spectrum 

### Part 2 - test of the real-world usability of peaks prediction within the UMSA Galaxy pipeline
TBD
