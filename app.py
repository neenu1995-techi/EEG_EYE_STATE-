import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

# Streamlit UI setup
st.title("🧠👀 DEEPEGAZE CLASSIFIER🧠👀")

# Upload dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the EEG dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Preprocessing
    df = df[df['Class'].isin([1, 2])].drop(columns=['id'], errors='ignore')
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = y - 1  # Convert labels to {0, 1}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM (Linear Kernel)": SVC(kernel='linear', probability=True),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    # Train and store results
    accuracy_results = {}
    roc_auc_results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model  # save the trained model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        accuracy_results[name] = acc
        roc_auc_results[name] = roc_auc

    # Dropdown for model selection
    st.sidebar.header("Select Model for Evaluation")
    selected_model_name = st.sidebar.selectbox("Choose a Model:", list(models.keys()))

    if selected_model_name:
        model = trained_models[selected_model_name]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Show Confusion Matrix
        st.subheader(f"Confusion Matrix - {selected_model_name}")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        # Show ROC Curve
        st.subheader(f"ROC Curve - {selected_model_name}")
        fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
        ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc_results[selected_model_name]:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend()
        ax_roc.grid()
        st.pyplot(fig_roc)

        st.success(f"✅ {selected_model_name} Accuracy: {accuracy_results[selected_model_name]:.4f}")

    # Final Results Button
    if st.button("Show Final Result 🎯"):
        st.subheader("Model Accuracy Comparison")
        fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(accuracy_results.values()), y=list(accuracy_results.keys()), palette="viridis", ax=ax_acc)
        ax_acc.set_xlabel('Accuracy')
        ax_acc.set_xlim(0, 1)
        ax_acc.set_title('Model Accuracy Comparison')
        ax_acc.grid(True)
        st.pyplot(fig_acc)

        best_model = max(accuracy_results, key=accuracy_results.get)
        best_accuracy = accuracy_results[best_model]
        st.success(f"🏆 Best Performing Model: {best_model} with Accuracy = {best_accuracy:.4f}")

else:
    st.info("👈 Please upload a CSV file to begin!")

