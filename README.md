# EEG EYE STATE 

## INTRODUCTION TO EEG

EEG stands for Electroencephalography. In simple terms: EEG is a method to record the electrical activity of the brain.

### Description about Dataset

All data is from one continuous EEG measurement with the Emotiv EEG Neuroheadset. The duration of the measurement was 117 seconds. The eye state was detected via a camera during the EEG measurement and added later manually to the file after analyzing the video frames. '1' indicates the eye-closed and '0' the eye-open state. All values are in chronological order with the first measured value at the top of the data.
The features correspond to 14 EEG measurements from the headset, originally labeled AF3, F7, F3, FC5, T7, P, O1, O2, P8, T8, FC6, F4, F8, AF4, in that order.

### EEG Data Machine Learning Pipeline

This repository contains a Jupyter Notebook that performs data preprocessing, feature engineering, and machine learning modeling on EEG (Electroencephalogram) data. The aim is to prepare the dataset for predictive modeling tasks.

## Project Structure
The notebook (Final_ML_EEG_DATA.ipynb) follows these main steps:

**1. Data Loading**
Importing EEG dataset(s) into the environment.

**2. Data Preparation**
Cleaning and organizing the dataset to prepare it for analysis.

**3. Summary Statistics**
Exploring and understanding the distribution of the data.

**4. Duplicate Check**
Identifying and removing any duplicate records.

**5. Feature Scaling and Normalization**
Applying normalization techniques to standardize feature ranges.

**6.Model Training (Likely in later sections)**
Applying machine learning models for classification or regression tasks based on the processed EEG data.

## Installation
To run this project locally:
Clone the repository or download the .ipynb file.
Set up a Python environment.

**1. Install the required packages:**

```
pip install -r requirements.txt
```
**2.If a requirements.txt is missing, install common libraries manually:**

```
pip install numpy pandas scikit-learn matplotlib seaborn
```

**3.Launch Jupyter Notebook:**
 ```
jupyter notebook
```


### Open Final_ML_EEG_DATA.ipynb to view and run the cells. 

## Requirements:
1. Python 3.7+
3. Jupyter Notebook
4. Libraries: numpy,pandas,scikit-learn,matplotlib,seaborn

(Additional libraries may be needed depending on later parts of the notebook.)

## Load the EEG dataset (instructions inside the notebook).
Follow the cells step-by-step:
 1. Data cleaning
 2. Feature scaling
 3. Model fitting
 4. Evaluation
 Modify parameters and models as needed for experiments.

# Streamlit App Overview for EEG Dataset
 This Streamlit app provides an interactive web interface to upload EEG datasets and run multiple machine learning models to classify EEG eye movement data.

 ## ✨ Features
Upload your EEG CSV dataset through a sidebar.

### 1. Preprocessing:

Filters dataset to include only Class 1 and Class 2.
Drops the 'id' column if present.
Standardizes features using StandardScaler.

### 2. Model Training:

Trains seven machine learning models automatically:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. SVM (Support Vector Machine) with linear kernel
6. Naive Bayes
7. Gradient Boosting

### 3.  Model Evaluation:

Allows users to select a model from a dropdown.
Shows the selected model's:
1. Confusion matrix (heatmap)
2. ROC Curve (with AUC score)
3. Accuracy Score

### 4. Final Comparison:

When clicking "Show Final Result 🎯", the app compares accuracies across all models using a bar chart.
Displays the best-performing model and its accuracy.

## 📦 Libraries Used
1. streamlit — for the interactive web UI
2. pandas,numpy — for data handling
4. seaborn,matplotlib — for visualization
5. scikit-learn — for preprocessing, model training, and evaluation

## 🛠 How It Works
User Uploads a CSV:
The app expects a dataset where one column is Class (for binary classification into 1 or 2).

### Automatic Processing:
The app handles scaling, label adjustment (changing {1,2} to {0,1}), and splitting into training and test sets.

### Training:
Each machine learning model is fit on the training data and evaluated on the test data.

### Interactive Visualization:
Choose any model to view detailed evaluation metrics.
View comparisons across all models easily.

## 🚀 To Run the Streamlit App Locally
**1. Make sure you have the necessary libraries installed:** 

     ''' pip install streamlit pandas numpy scikit-learn seaborn matplotlib'''

**2.Run the app:**

      ''' streamlit run app.py'''

**3.Upload your EEG CSV file and start exploring!**

### 🧠 Important Notes
Input Format:
The uploaded CSV must have a Class column and (optionally) an id column.

**Binary Classification:**
Only samples where Class is 1 or 2 are considered.

**Scaling:**
StandardScaler is applied to features to normalize the dataset before modeling.













