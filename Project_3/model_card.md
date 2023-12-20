# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Decision Tree model were trained.

* Model version: 1.0.0
* Model date: 21 December 2023

## Intended Use
The model is used to predict the income classes on census data based on various information. There are two income classes, which are >50K and <=50K, respectively (this is considered as a binary classification task).

## Training Data
The UCI Census Income Data Set was used for training. Further information on the dataset can be found at https://archive.ics.uci.edu/ml/datasets/census+income
For training 80% of the 32561 rows were used (26561 instances) in the training set.

## Evaluation Data
For evaluation 20% of the 32561 rows were used (6513 instances) in the test set.

## Metrics
Three metrics were used for model evaluation (these value are measured on the test set):
* precision: 0.7028938906752411
* recall: 0.6713759213759214
* f1: 0.6867734841344644

## Ethical Considerations
Since the dataset consists of public available data with highly aggregated census data no harmful unintended use of the data has to be addressed.

## Caveats and Recommendations
It would be meaningful to either perform an hyperparameter optimization or try other models (e.g. ensemble models) to improve the metric results.
