# EEG EYE STATE 
# 🧠 DEEPEGAZE CLASSIFIER 👀

This Streamlit web app performs binary classification on EEG eye-tracking data using multiple supervised machine learning algorithms. It enables easy comparison of model performance, evaluation through confusion matrix and ROC curves, and provides visual insights into the results.The goal is to process raw EEG EYE STATE signals and build predictive models that can classify cognitive or neurological states when the eyes is open and closed.

---



### Description about Dataset

All data is from one continuous EEG measurement with the Emotiv EEG Neuroheadset. The duration of the measurement was 117 seconds. The eye state was detected via a camera during the EEG measurement and added later manually to the file after analyzing the video frames. '1' indicates the eye-closed and '0' the eye-open state. All values are in chronological order with the first measured value at the top of the data.
The features correspond to 14 EEG measurements from the headset, originally labeled AF3, F7, F3, FC5, T7, P, O1, O2, P8, T8, FC6, F4, F8, AF4, in that order.

## 📁 Project Structure

---

## 🧪 ML Pipeline

1. **Data Collection**  
   - Uses `EEG_EYE_DATASET.csv` with EEG eye-tracking data and class labels.

2. **Data Cleaning & Preprocessing**  
   - Drops irrelevant columns like `id`  
   - Keeps only binary class labels (1 and 2 → converted to 0 and 1)  
   - Scales features using `StandardScaler`

3. **Exploratory Data Analysis (EDA)**  
   - Displays dataset preview  
   - Heatmaps for confusion matrices  
   - Accuracy bar chart for model comparison

4. **Model Building**  
   Implements 7 supervised classification algorithms:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - Naive Bayes
   - Gradient Boosting

5. **Model Evaluation**  
   - Accuracy Score  
   - Confusion Matrix (Seaborn Heatmap)  
   - ROC Curve & AUC Score

6. **Model Saving & Deployment**  
   - Real-time interactive web app using Streamlit  
   - Deployment-ready (locally or on Streamlit Cloud)

---

## 🧰 Tech Stack / Tools Used

- **Language**: Python  
- **Data Analysis**: Pandas, NumPy  
- **Machine Learning**: Scikit-learn  
- **Visualization**: Matplotlib, Seaborn, Plotly  
- **Web App**: Streamlit  
- **Environment**: Jupyter Notebook & Python Scripts

---

## 📊 Results

- Evaluation Metrics (varies per model):
  - **Accuracy**
  - **ROC AUC Score**
  - **Confusion Matrix**
- Final output displays a **comparison of accuracy** across models and highlights the best performer.

---

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

# Streamlit App Overview for EEG Dataset
 This Streamlit app provides an interactive web interface to upload EEG datasets and run multiple machine learning models to classify EEG eye movement data.


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


# Authors 
NEENU P RAMACHANDRAN










