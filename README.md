# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

Heart Disease Prediction is a machine learning classification project that predicts whether a patient is likely to have heart disease based on various clinical and medical attributes.

The project implements a complete machine learning pipeline starting from data loading and preprocessing to model training, evaluation, visualization, model serialization, and deployment through an interactive Streamlit web application.

The project uses the **UCI Heart Disease dataset**, containing **920 patient records** and multiple clinical features such as age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate, exercise-induced angina, ST depression, slope, number of major vessels, and thalassemia.

Two machine learning algorithms are implemented and compared:

1. **Logistic Regression**
2. **Decision Tree Classifier**

Logistic Regression is used as the primary prediction model, while Decision Tree is used as a comparison model.

The final trained Logistic Regression model is integrated into a Streamlit application where users can enter patient information and receive a predicted likelihood of heart disease.

> **Important:** This project is intended for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used to make real-world medical decisions.

---

# 🎯 Project Objectives

The main objectives of this project are:

- Understand and implement a complete machine learning pipeline.
- Perform data cleaning and preprocessing.
- Handle missing numerical and categorical values.
- Convert categorical variables into machine-readable features.
- Apply feature scaling to numerical variables.
- Convert the original multi-class heart disease target into a binary classification problem.
- Train multiple classification algorithms.
- Compare model performance.
- Evaluate models using multiple classification metrics.
- Generate a confusion matrix.
- Generate and analyze ROC curves.
- Calculate ROC-AUC scores.
- Save trained machine learning models.
- Build an interactive Streamlit web application.
- Allow users to enter patient information and receive a prediction.
- Document the complete project for academic and portfolio purposes.

---

# 🩺 Problem Statement

Heart disease is one of the major health challenges worldwide. Early identification of patients who may be at higher risk can potentially support further medical evaluation.

Machine learning can be used to identify patterns in historical medical data and build predictive models.

The goal of this project is to develop a binary classification system that predicts:

```text
0 → Lower likelihood / No heart disease
1 → Higher likelihood / Heart disease
```

based on patient clinical attributes.

The model learns patterns from historical patient records and uses those patterns to classify new input data.

---

# 💡 Proposed Solution

The proposed system follows a complete machine learning workflow:

```text
                 Heart Disease Dataset
                          │
                          ▼
                   Data Collection
                          │
                          ▼
                    Data Cleaning
                          │
                          ▼
                 Missing Value Handling
                          │
                          ▼
                 Feature Engineering
                          │
                          ▼
                 Target Transformation
                          │
                          ▼
                Train/Test Data Split
                          │
                          ▼
                Feature Preprocessing
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       Logistic Regression       Decision Tree
             │                         │
             └────────────┬────────────┘
                          ▼
                    Model Evaluation
                          │
             ┌────────────┼─────────────┐
             ▼            ▼             ▼
          Accuracy     Precision      Recall
             │            │             │
             └────────────┼─────────────┘
                          ▼
                    F1 Score / ROC-AUC
                          │
                          ▼
                   Model Comparison
                          │
                          ▼
                   Model Serialization
                          │
                          ▼
                 Streamlit Web Application
                          │
                          ▼
                    Patient Prediction
```

---

# 📊 Dataset Information

## Dataset Name

**Heart Disease Dataset**

The dataset is based on the UCI Heart Disease dataset and contains clinical information collected from patients.

The dataset used in this project contains:

```text
Number of Rows: 920
Number of Columns: 16
```

The dataset includes numerical, categorical, and binary attributes.

---

# 📋 Dataset Columns

The original dataset contains the following columns:

| Column | Description | Type |
|---|---|---|
| `id` | Unique patient record identifier | Integer |
| `age` | Age of the patient | Integer |
| `sex` | Biological sex of the patient | Categorical |
| `dataset` | Dataset/source location | Categorical |
| `cp` | Chest pain type | Categorical |
| `trestbps` | Resting blood pressure | Numerical |
| `chol` | Serum cholesterol level | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting electrocardiographic result | Categorical |
| `thalch` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise-induced angina | Binary |
| `oldpeak` | ST depression induced by exercise | Numerical |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels | Numerical |
| `thal` | Thalassemia information | Categorical |
| `num` | Original heart disease diagnosis target | Integer |

---

# 🎯 Target Variable

The original dataset uses:

```text
num
```

as the target variable.

The values represent different levels of heart disease.

For this project, the target is transformed into a binary classification target called:

```text
target
```

The transformation is:

```python
df["target"] = df["num"].apply(
    lambda x: 1 if x > 0 else 0
)
```

Therefore:

```text
num = 0
       ↓
target = 0
       ↓
No Heart Disease

num > 0
       ↓
target = 1
       ↓
Heart Disease
```

This converts the original target into a binary classification problem.

---

# 🧹 Data Preprocessing

Data preprocessing is one of the most important stages of this project.

The raw dataset contains missing values, categorical values, and columns that should not be used directly by the machine learning model.

The following preprocessing steps are performed.

---

## 1. Duplicate Removal

Duplicate records are removed using:

```python
df = df.drop_duplicates()
```

This prevents duplicate observations from unnecessarily influencing the model.

---

## 2. Target Transformation

The original `num` column is transformed into a binary target:

```python
df["target"] = df["num"].apply(
    lambda x: 1 if x > 0 else 0
)
```

After transformation, the original `num` column is removed.

---

## 3. ID Removal

The `id` column is removed because it is simply an identifier and does not provide meaningful information about whether a patient has heart disease.

```python
df = df.drop("id", axis=1)
```

Including identifiers in a machine learning model can introduce meaningless patterns.

---

## 4. Dataset Column Removal

The `dataset` column identifies the source/location of the record.

It is not considered a meaningful clinical predictor for this project, so it is removed.

```python
df = df.drop("dataset", axis=1)
```

---

# 🕳️ Missing Value Handling

The dataset contains missing values in several columns.

Examples include:

```text
trestbps
chol
fbs
thalch
exang
oldpeak
slope
ca
thal
```

Instead of simply deleting all rows containing missing values, the project uses appropriate imputation strategies.

---

## Numerical Features

Numerical missing values are replaced using the **median**.

```python
SimpleImputer(strategy="median")
```

Median imputation is useful because it is less sensitive to extreme values than mean imputation.

---

## Categorical Features

Categorical missing values are replaced using the **most frequently occurring value**.

```python
SimpleImputer(
    strategy="most_frequent"
)
```

This allows the model to process incomplete categorical records without deleting potentially useful patient observations.

---

# 🔤 Categorical Feature Encoding

Several features contain text values.

Examples:

```text
sex
cp
restecg
slope
thal
```

Machine learning algorithms require numerical representations of categorical variables.

Therefore, categorical features are converted using:

```python
OneHotEncoder(
    handle_unknown="ignore"
)
```

For example:

```text
sex

Male
Female
```

can be represented numerically through one-hot encoding.

The `handle_unknown="ignore"` option prevents the application from crashing if an unexpected category appears during prediction.

---

# 📏 Feature Scaling

Numerical features can have significantly different ranges.

For example:

```text
Age              → approximately 30–80
Cholesterol      → hundreds
Oldpeak          → usually small decimal values
Maximum HR       → approximately 70–200
```

Therefore, numerical features are standardized using:

```python
StandardScaler()
```

The scaler is fitted only on the training data and then applied to the test data.

This prevents information from the test set from leaking into the training process.

---

# 🔀 Train-Test Split

The processed dataset is divided into training and testing datasets.

The project uses:

```text
80% → Training
20% → Testing
```

The split is performed using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Why `random_state=42`?

It makes the experiment reproducible.

Running the same code again with the same dataset and configuration produces the same train/test split.

### Why `stratify=y`?

It helps preserve the class distribution between the training and testing datasets.

---

# 🤖 Machine Learning Models

Two classification algorithms are implemented.

---

# 1. Logistic Regression

Logistic Regression is used as the primary model.

It is a widely used algorithm for binary classification problems.

In this project, it predicts:

```text
0 → No/Less likelihood of heart disease
1 → Heart disease
```

The implementation uses:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### Advantages

- Simple and interpretable.
- Efficient for binary classification.
- Works well with scaled features.
- Produces probability estimates.
- Suitable as a strong baseline model.

---

# 2. Decision Tree Classifier

The second model is a Decision Tree classifier.

The model learns decision rules from the training data.

For example, a simplified decision process could conceptually look like:

```text
             Age > threshold?
                  │
          ┌───────┴───────┐
         Yes              No
          │                │
     Cholesterol?       Heart rate?
          │                │
        ...              ...
```

The project uses:

```python
DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
```

### Advantages

- Easy to understand.
- Can model non-linear relationships.
- Does not require feature scaling.
- Useful for comparing against Logistic Regression.

---

# 📈 Model Evaluation

The models are evaluated using several metrics.

---

## Accuracy

Accuracy measures the percentage of total predictions that are correct.

```text
Accuracy =
Correct Predictions / Total Predictions
```

A higher accuracy generally indicates that the model is correctly classifying more observations.

However, accuracy alone should not be used to judge a medical classification model.

---

# Precision

Precision answers:

> Of the patients predicted as having heart disease, how many actually had heart disease?

```text
Precision =
True Positives /
(True Positives + False Positives)
```

High precision means the model generates fewer false positive predictions.

---

# Recall

Recall answers:

> Of the patients who actually had heart disease, how many did the model correctly identify?

```text
Recall =
True Positives /
(True Positives + False Negatives)
```

Recall is especially important in medical screening problems because missing a positive case can be more concerning than generating an additional false positive.

---

# F1 Score

F1 Score combines precision and recall.

```text
F1 =
2 × Precision × Recall /
(Precision + Recall)
```

A higher F1 score indicates a better balance between precision and recall.

---

# 🔲 Confusion Matrix

The project generates a confusion matrix for Logistic Regression.

A binary confusion matrix contains:

```text
                    Predicted
                 Negative Positive

Actual Negative     TN       FP

Actual Positive     FN       TP
```

Where:

- **TN** = True Negative
- **TP** = True Positive
- **FP** = False Positive
- **FN** = False Negative

The generated visualization is saved as:

```text
outputs/confusion_matrix.png
```

---

# 📉 ROC Curve

The project also generates a Receiver Operating Characteristic curve.

The ROC curve compares:

```text
True Positive Rate
```

against:

```text
False Positive Rate
```

at different classification thresholds.

The ROC curve helps evaluate how well the model separates the two classes.

The generated graph is saved as:

```text
outputs/roc_curve.png
```

---

# 🏆 ROC-AUC

The Area Under the ROC Curve (AUC) provides a single-value summary of the model's ability to distinguish between the two classes.

An AUC closer to:

```text
1.0
```

generally indicates better discrimination.

An AUC around:

```text
0.5
```

indicates performance close to random classification.

The project calculates AUC for both:

```text
Logistic Regression
Decision Tree
```

---

# 📊 Model Comparison

The project compares the two models using their accuracy.

The comparison visualization is saved as:

```text
outputs/model_comparison.png
```

The project evaluates:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | Calculated during training | Calculated during training | Calculated during training | Calculated during training | Calculated during training |
| Decision Tree | Calculated during training | Calculated during training | Calculated during training | Calculated during training | Calculated during training |

> The actual values should be taken from the output produced by `train_model.py` rather than hardcoded into the README.

---

# 💾 Model Serialization

After training, the project saves the trained models using `joblib`.

The following files are generated:

```text
models/
├── logistic_model.pkl
├── decision_tree_model.pkl
└── preprocessor.pkl
```

### `logistic_model.pkl`

Contains the trained Logistic Regression model.

### `decision_tree_model.pkl`

Contains the trained Decision Tree model.

### `preprocessor.pkl`

Contains the complete preprocessing pipeline, including:

- Numerical missing-value imputation
- Numerical feature scaling
- Categorical missing-value imputation
- One-hot encoding

Saving the preprocessing pipeline is important because the exact same transformations must be applied to new patient data during prediction.

---

# 🌐 Streamlit Web Application

The project includes an interactive Streamlit application.

The application allows users to enter patient information through a web interface.

The application accepts:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- Oldpeak
- Slope
- Number of major vessels
- Thalassemia

After entering the information, the user can click:

```text
Predict Heart Disease
```

The application then:

1. Creates a patient data record.
2. Applies the saved preprocessing pipeline.
3. Sends the processed data to the Logistic Regression model.
4. Generates a prediction.
5. Calculates the predicted probability.
6. Displays the result.

---

# 🖥️ Application Output

The application produces one of two primary