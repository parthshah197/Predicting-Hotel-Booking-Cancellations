# Predicting Hotel Booking Cancellations

## Table of Contents
- [Project Overview](#Project-Overview)
- [Data Description](#Data-Description)
- [Feature Engineering](#Feature-Engineering)
- [Preprocessing Pipeline](#Preprocessing-Pipeline)
- [Modeling](#Modeling)  
  - [Model Evaluation](#Model-Evaluation)
  - [Hyperparameter Optimization](#Hyperparameter-Optimization)
    - [Random Search Cross Validation](#Random-Search-Cross-Validation)
    - [Bayesian Optimization](#Bayesian-Optimization)
  - [Ensemble: Stacking Classifier](#Ensemble--Stacking-Classifier)
- [Model Interpretation using SHAP](#Model-Interpretation-using-SHAP)

## Project Overview
This project leverages Scikit-Learn’s pipeline for data preprocessing and hyperparameter tuning of models (Random Forest, LightGBM, Logistic Regression, XGBoost, CatBoost) using Random Search Cross Validation and Bayesian Optimization to predict hotel booking cancellations. A stacked ensemble approach with a logistic regression meta-classifier is utilized to combine predictions from multiple base models, achieving an F1 score of 0.86. Feature importance is interpreted and model predictions are explained using SHAP values.

## Data Description
The dataset contains various features related to hotel bookings. Key features include:
- `hotel`: Type of hotel (Resort Hotel or City Hotel)
- `is_canceled`: Booking cancellation status (1 if canceled, 0 otherwise)
- `lead_time`: Number of days between the booking date and the arrival date
- `arrival_date_year`: Year of arrival date
- `arrival_date_month`: Month of arrival date
- `country`: Country of origin
- `market_segment`: Market segment designation
- `distribution_channel`: Booking distribution channel

## Feature Engineering
New features are created to enhance the model's predictive power. Notable features include:
- `country_ratio`: Cancellation ratio at country level
- `farbook`: Indicator of whether the booking was made well in advance (lead time > 7 days)

## Preprocessing Pipeline
To streamline the preprocessing and model training processes, we leveraged scikit-learn’s `Pipeline` and `ColumnTransformer`. This allowed for seamless integration of various preprocessing steps and models, ensuring a robust and scalable workflow.

### Scikit-Learn Pipeline
The preprocessing pipeline included several key steps:
1. **Imputation**: Handling missing values using `SimpleImputer`.
2. **Encoding**: Converting categorical variables using `OneHotEncoder`.
3. **Scaling**: Standardizing numerical features using `StandardScaler`.
4. **Custom Transformation**: Creating a custom transformer class for specific feature engineering tasks.

## Modeling
Various machine learning models are trained to predict booking cancellations. The models used include:
- Logistic Regression
- Random Forest
- SGDClassifier
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- XGBoost
- LightGBM
- CatBoost
- AdaBoost
- Stacking Classifier

### Model Evaluation
Models were evaluated using accuracy, F1 score and classification reports. Random Forest, XGBoost, and CatBoost demonstrated the highest accuracies & F1 score.

### Hyperparameter Optimization
Hyperparameter tuning was performed using both Random Search Cross Validation (CV) and Bayesian Optimization. These techniques were essential in finding the best parameters for the models to enhance their predictive performance.

#### Random Search Cross Validation
Random Search CV was used to explore a wide range of hyperparameter values in a computationally efficient manner. It allows for random sampling of hyperparameter values from a predefined grid, providing a balance between exhaustive search and speed.

#### Bayesian Optimization
Bayesian Optimization, implemented using `BayesSearchCV` from `scikit-optimize`, provided a more efficient hyperparameter search by building a probabilistic model of the objective function. This method focuses on promising areas of the hyperparameter space, leading to faster convergence to optimal values.

### Ensemble: Stacking Classifier
A stacked ensemble approach was utilized to combine predictions from multiple base models. The base models used included Random Forest, XGBoost, and LightGBM. A logistic regression meta-classifier was employed to make the final predictions based on the outputs of these base models. This approach leverages the strengths of different models (that were optimized using Bayesian optimization), leading to improved overall performance and achieving an **F1 score = 0.86**.

## Model Interpretation using SHAP
SHAP (SHapley Additive exPlanations) was used to interpret the feature importance and explain the model predictions. SHAP values provide insights into how each feature contributes to the final prediction. The key insights include:

![SHAP](https://github.com/user-attachments/assets/a7359853-020f-40de-a3dc-90adc1be0344)

- **Country**: The origin of the booking can influence cancellation rates, with certain countries having higher cancellation ratios.
- **Lead Time**: Longer lead times generally increase the likelihood of cancellation.
- **Deposit Type**: Type of deposit (no deposit, refundable, non-refundable) while booking severly impacts the probability of cancellation.
- **Previous history**: Customers with previous cancellations are more likely to cancel a reservation.




