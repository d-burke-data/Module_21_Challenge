# Module_21_Challenge : *Deep Learning*
This is a demonstration of a neural network using TensorFlow and KerasTuner in Python/Jupyter Notebook. The Jupyter Notebook files were executed in Google Colab.

## Repository Directory
|File|Description|
|---|---|
|AlphabetSoupCharity.h5|HDF5 model file from initial neural network|
|AlphabetSoupCharity_Optimization.h5|HDF5 model file from optimized neural network|
|DeepLearning.ipynb|Initial deep learning execution|
|DeepLearning_Optimized.ipynb|Attempted manual and KerasTuner optimizations|

# Analysis
## Overview
The purpose of this analysis is to find an algorithm to determine which applicants would be most likely to succeed if granted funding by Alphabet Soup, based on several descriptive features.
### Data
The data provided is 34,299 rows in CSV format with the following columns:
|Column|Type|Description|
|---|---|---|
|EIN|Categorical|Employer Identification Number|
|NAME|Categorical|Organization name|
|APPLICATION_TYPE|Categorical|Alphabet Soup application type|
|AFFILIATION|Categorical|Affiliated sector of industry|
|CLASSIFICATION|Categorical|Government organization classification|
|USE_CASE|Categorical|Use case for funding|
|ORGANIZATION|Categorical|Organization type|
|STATUS|Categorical|Active status|
|INCOME_AMT|Categorical|Bins of income classification|
|SPECIAL_CONSIDERATIONS|Categorical|Special considerations for application|
|ASK_AMT|Numeric|Funding amount requested|
|IS_SUCCESSFUL|Categorical|Was the money used effectively?|

## Results
### Data Preprocessing
1. The target variable is `IS_SUCCESSFUL`, a binary categorical column (1 = successful, 0 = not successful)
2. The features are: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, and `ASK_AMT`.
3. The identification columns `EIN` and `NAME` were dropped as they have no bearing on the analysis. 

### Compiling, Training, and Evaluating the Model
1. The recommended starting hyperparameters were:
   
   |Hyperparameter|Value(s)|
   |---|---|
   |Activation Function|ReLU|
   |Number of Layers|2|
   |Neuron Units By Layer|80, 30|
   |Output layer Activation Function|Sigmoid|

&emsp;&emsp;Sigmoid was used for the outer layer activation function since we are looking for a binary outcome (successful or not).

2. This model achieved 0.7350 validation accuracy, short of the 0.75 target value.
3. The following steps were implemented to attempt to improve the model's performance:
   + Removed variables `STATUS` and `SPECIAL_CONSIDERATIONS`, as these are binary categories with a single value assigned to the overwhelming majority (>99.9%) of data points.
   + Condensed variables `AFFILIATION`, `USE_CASE`, and `ORGANIZATION` to merge low-count values into an 'Other' category, as previously done with `APPLICATION_TYPE` and `CLASSIFICATION`.
   + Used KerasTuner to automate optimizing hyperparameters for the model, with activation functions ReLU, Leaky ReLU, Tanh, and Mish. Neuron units per layer ranged from 10 to 100 in 10-step interals, with 2 to 7 layers before the output layer. Two different tuners were used, Hyperband and BayesianOptimization.
   + Neuron units per layer were modified on subsequent tunings to narrow the range based on previous results.
   + Ultimately, even with these optimization attempts, the model's validation accuracy did not achieve the desired target. The best result was a BayesianOptimization Tuner with the following hyperparameters, with an accuracy of 0.7406:

Hyperparameters (Top 3 Models):  
![Optimization Results Hyperparameters](https://github.com/user-attachments/assets/f8831206-1971-4ed5-8b07-d8dec48f563e)  
Accuracy (Top 3 Models):  
![Optimization Results Accuracy](https://github.com/user-attachments/assets/fc7e1eb1-7e47-4aa6-9f5d-5c827740235f)

## Summary
Running the model through KerasTuner multiple times with different hyperparameter settings, features setups, and tuners achieved very similar results to the initial settings, none of which met the desired accuracy of 0.75. While the options used were by no means exhaustive, it is possible that the data itself may be insufficient to reach the desired outcome. A dataset with additional columns providing more information about each applicant organization may be helpful to achieve better accuracy. Since that may not be possible, expanding the KerasTuner options to include a greater range of activation functions, neurons per layer, number of layers, and number of epochs may yield better results at the cost of additional time.
