<h1 align="center">
  HeartBeat
</h1>
<p align="center">
  <img src="img/HeartBeat.gif" alt="HeartBeatGif">
</p>

# Table of contents
1. [Introduction](#introduction)
2. [Main Features](#main-features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Data Source and Structure](#data-source-and-structure)
7. [Validation](#validation)
8. [Licence](#license)

## Introduction

Get your heart racing with HeartBeat! This AI-powered machine learning project skips a beat to predict heart disease. It's like a cardio workout for your data, crunching numbers to keep your health in check. The system utilizes different machine learning models to make predictions and provides accuracy metrics for evaluation.

## Main Features
- Utilizes various machine learning models for predicting heart disease.
- Computes the average age of patients present in the dataset.
- Offers accuracy metrics and classification reports for evaluating each model's performance.

Additionally, the project includes:
- [x] Use of basic Python data structures:
    - [x] Lists
    - [x] Tuples
    - [x] Dictionaries
- [x] Implementation of a Class with:
    - [x] Inheritance
    - [x] Attribute lookups in objects/classes
- [x] Use of Python descriptors or magic methods
- [x] Use of at least one decorator
- [x] Other required project elements:
    - [x] Variable assignments
    - [x] Use of methods
    - [x] Creation/invocation of functions
    - [x] Use of loops

## Requirements
### For installation
[python-pip](https://pip.pypa.io/en/stable/) - pip is the package installer for Python, and it allows you to install packages from various sources, such as the Python Package Index.

The following software, libraries, and versions are required to run the project:

- Python (version 3.8 or higher)
- pandas
- numpy
- scikit-learn
- plotly

## Installation

To install and set up the project, follow these steps:

1. Install Python (version 3.8 or higher) on your system.
2. Open a command prompt or terminal.
3. Clone the project repository.
4. Navigate to the project directory.
5. Create a virtual environment (optional but recommended).
6. Install the required libraries by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

To utilize the project, adhere to the following steps:

1. Procure the dataset for heart disease prediction. The dataset needs to be in CSV format.
2. Position the dataset file in the same directory as the project code.
3. If necessary, modify the code to indicate the appropriate dataset filename.
4. Execute the project code using Python.
5. The program will then train the models on the dataset and formulate predictions.
6. The performance accuracy of each model and its respective classification report will be displayed.
7. The program will also compute and display the average age of patients present in the dataset.

Example of how to use:

```bash
python3 main.py

```

## Data Source and Structure
The data used in this project is sourced from a CSV file named "heart.csv". It contains various attributes of patients related to heart disease, including age, sex, chest pain type, resting blood pressure, cholesterol levels, and more.

The dataset has the following structure:

| Attribute            | Description                           |
|----------------------|---------------------------------------|
| age                  | age in years                   |
| sex                  | sex of the patient (0 = female, 1 = male) |
| cp                   | Type of chest pain                     |
| trestbps             | resting blood pressure (in mm Hg on admission to the hospital)|
| chol                 | serum cholesterol in mg/dl   |
| fbs                  | (fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false) |
| restecg              | resting electrocardiographic results    |
| thalach              | maximum heart rate achieved             |
| exang                | exercise induced angina (1 = yes; 0 = no)               |
| oldpeak              | ST depression induced by exercise relative to rest      |
| slope                | Slope of the peak exercise ST segment   |
| ca                   | Number of major vessels (0-3) coloured by fluoroscopy|
| thal                 | Thalassemia 0 = normal; 1 = fixed defect; 2 = reversible defect |
| target               | Presence of heart disease (0 = no, 1 = yes) |

## Validation
The project performs model validation by splitting the dataset into training and testing sets. The models are trained on the training set and then tested on the testing set to evaluate their accuracy.

The following machine learning models are used for heart disease prediction:

- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Decision Tree
- Support Vector Machine

For each model, the accuracy score and classification report are displayed, providing insights into the model's performance.

Additionally, the project calculates and displays the average age of patients in the dataset.

After starting the project using,
```bash
python3 main.py
```
we can notice that each model has different accuracy. So, here is an explanation of why this is happening with outputs from each model: 

Each model's accuracy can be attributed to several factors, such as the algorithms they use, how they are built, and how well they fit the given dataset. Let's discuss the potential reasons for the variation in accuracies:

- Random Forest:

   - Random Forest is an ensemble method that combines multiple decision trees. It tends to perform well on a wide range of datasets.
   - The accuracy of 0.985 indicates that the Random Forest model achieved a very high level of accuracy on the test data.
   - Random Forests are known for their ability to handle complex relationships and noisy data, which might contribute to their high accuracy in this case.
   
Random Forest output:
   
```console
Model 1 - RandomForestClassifier Accuracy: 0.985
Model 1 - RandomForestClassifier Classification Report:
              precision    recall  f1-score   support

           0       0.97      1.00      0.99       102
           1       1.00      0.97      0.99       103

    accuracy                           0.99       205
   macro avg       0.99      0.99      0.99       205
weighted avg       0.99      0.99      0.99       205
```

- Gradient Boosting:

    - Gradient Boosting is an ensemble method that combines weak learners, usually decision trees, to make predictions.
    - An accuracy of 0.932 suggests a good performance by the Gradient Boosting model, although not as high as Random Forest.
    - Gradient Boosting models can be powerful but might require careful tuning of hyperparameters to achieve optimal performance.

Gradient Boosting output:

```console
Model 2 - GradientBoostingClassifier Accuracy: 0.932
Model 2 - GradientBoostingClassifier Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.91      0.93       102
           1       0.92      0.95      0.93       103

    accuracy                           0.93       205
   macro avg       0.93      0.93      0.93       205
weighted avg       0.93      0.93      0.93       205
```

- K-Nearest Neighbors (KNN):

  - KNN is a non-parametric algorithm that assigns a class to a sample based on the majority class of its nearest neighbors.
  - An accuracy of 0.732 indicates relatively lower performance compared to other models.
  - KNN can struggle with high-dimensional data or when the dataset has imbalanced class distributions.
  
KNN output:

```console
Model 3 - KNeighborsClassifier Accuracy: 0.732
Model 3 - KNeighborsClassifier Classification Report:
              precision    recall  f1-score   support

           0       0.73      0.73      0.73       102
           1       0.73      0.74      0.73       103

    accuracy                           0.73       205
   macro avg       0.73      0.73      0.73       205
weighted avg       0.73      0.73      0.73       205
```

- Decision Tree:

   - Decision Tree is a model that uses a tree-like structure to make decisions based on feature values.
   - An accuracy of 0.985 suggests that the Decision Tree model performed very well on the given data.
   - Decision Trees can capture complex relationships in the data, but might be prone to overfitting if not pruned or regularized.
   
Decision Tree output:
   
```console
Model 4 - DecisionTreeClassifier Accuracy: 0.985
Model 4 - DecisionTreeClassifier Classification Report:
              precision    recall  f1-score   support

           0       0.97      1.00      0.99       102
           1       1.00      0.97      0.99       103

    accuracy                           0.99       205
   macro avg       0.99      0.99      0.99       205
weighted avg       0.99      0.99      0.99       205
```

- Support Vector Machine (SVM):

  - SVM is a powerful algorithm that constructs hyperplanes to separate data points in high-dimensional space.
  - An accuracy of 0.683 indicates relatively lower performance compared to other models.
  - SVM's performance can be sensitive to the choice of kernel, regularization parameters, and data scaling. It might struggle with complex or overlapping class boundaries.
  
SVM output:

```console  
Model 5 - SVC Accuracy: 0.683
Model 5 - SVC Classification Report:
              precision    recall  f1-score   support

           0       0.71      0.61      0.66       102
           1       0.66      0.76      0.71       103

    accuracy                           0.68       205
   macro avg       0.69      0.68      0.68       205
weighted avg       0.69      0.68      0.68       205
```

It's important to note that the dataset itself and its characteristics, such as the distribution of classes, feature relationships, and noise, can also influence the model's performance. Additionally, the choice of hyperparameters and the training/validation data split can impact the accuracies. It is common to experiment with different models, hyperparameters, and evaluation metrics to identify the best-performing model for a specific task.

## Explaining the components in the classification report:

1. Precision:

    - Precision is the proportion of true positive predictions out of all positive predictions made by the model.
    - It measures the model's ability to avoid false positives, i.e., correctly identifying the positive class.
    - A high precision indicates that the model has a low rate of false positives.
    
2. Recall (also known as sensitivity or true positive rate):

    - Recall is the proportion of true positive predictions out of all actual positive samples in the dataset.
    - It measures the model's ability to find all positive samples and avoid false negatives.
    - A high recall indicates that the model has a low rate of false negatives.
3. F1-score:

    - The F1-score is the harmonic mean of precision and recall.
    - It provides a single metric that balances both precision and recall.
    - The F1-score is useful when there is an imbalance between the classes in the dataset.
4. Support:

    - Support represents the number of samples in each class in the dataset.
    - It provides insight into the distribution of samples across different classes.
    -The support value can help identify potential class imbalances or biases in the dataset.
    
A classification report typically presents these metrics for each class in the target variable. It allows assessing the model's performance across different classes, which is particularly useful when dealing with multi-class classification problems.

Example of a binary classification report (not related to the provided CSV file) to illustrate how the metrics are interpreted:

```console  
              precision    recall  f1-score   support

    Class 0       0.80      0.90      0.85       100
    Class 1       0.75      0.60      0.67        50

   micro avg       0.78      0.78      0.78       150
   macro avg       0.77      0.75      0.76       150
weighted avg       0.78      0.78      0.77       150

```
For Class 0:

- Precision: 80% of the positive predictions for Class 0 were correct.
Recall: The model identified 90% of the actual Class 0 samples.
- F1-score: The harmonic mean of precision and recall for Class 0 is 85%.
- Support: There are 100 samples belonging to Class 0 in the dataset.

For Class 1:

- Precision: 75% of the positive predictions for Class 1 were correct.
- Recall: The model identified 60% of the actual Class 1 samples.
- F1-score: The harmonic mean of precision and recall for Class 1 is 67%.
- Support: There are 50 samples belonging to Class 1 in the dataset.

## Licence

![MIT Image](https://upload.wikimedia.org/wikipedia/commons/0/0c/MIT_logo.svg)

Software licensed under the MIT Licence.