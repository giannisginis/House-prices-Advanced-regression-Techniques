# House-prices-Advanced-regression-Techniques

## Introduction
In this Kaggle challenge, we use different Regression and other Machine Learning techniques to predict the final price for each home using the features provided in the dataset. Before running ML algorithms, we preprocessed the data to remove outliers, to deal with missing values, and to encode categorical variables.

## Data Preprocessing and Exploration
### In the eda section we do the following:

* Load the csv files
* 1st dataset → One-hot-encoding to categorical features
   *Cons: Curse of Dimensionality
* 2nd dataset → Map categorical values to corresponding numerical
   * Standardization
   * Null values
   * Delete columns with more than 15% null values
   * Fill those left with median value.
* Correlation with SalePrice
   * Chose features with good correlation
   * Correlation ∈[−1,−0.4]∪[0.48,1]
* Between high correlated features keep the important ones

## Model Selection
We decided to use Ridge,Lasso,Linear and Support Vector Machine Regression, Gradient Boosting Machine (GradientBoostingRegressor), Random Forest and ExtraTreesRegressor which are all part of sklearn package to predict the output and finally a deep learning approach by using keras package with tensoflow as backend. All the algorithms had similar root mean square logarithmic error while Deep learning model showed the best score.

## Model Learning
We can improve a model’s performance by tuning its parameters. So we used grid search method to tune hyperparameters for lasso and Ridge regressor, GradientBoostingRegressor, Random Forest and ExtraTreesRegressor.
