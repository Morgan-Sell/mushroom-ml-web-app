import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

    @st.cache(max_entries=10) # caches output to disk. prevents reruning the function.
    def load_data():
        data = pd.read_csv("mushrooms.csv")
        label = LabelEncoder()

        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache(max_entries=10)
    def split(df):
        y = df["type"]
        X = df.drop(columns=["type"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        return X_train, X_test, y_train, y_test
    

    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            plot_confusion_matrix(model, X_test, y_test, display_labels=class_names, ax=ax)
            st.pyplot(fig)
        
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            fig, ax =plt.subplots()
            plot_roc_curve(model, X_test, y_test, ax=ax)
            st.pyplot(fig)
        
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            plot_precision_recall_curve(model, X_test, y_test, ax=ax)
            st.pyplot(fig)
    

    def  calc_and_plot_metrics(model, X_train, X_test, y_train, y_test):
        """ Calculates classification model metrics.
        """
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)

        st.write("Accuracy: ", accuracy.round(3))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(3))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(3))
        plot_metrics(metrics)

    # Body of "main" code.
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    class_names = ["edible", "poisonous"]
    
    st.sidebar.subheader("Choose Classification Model")
    classifier = st.sidebar.selectbox("Select Classifier", ("Support Vector Machine (SVM)",
                                                            "Logistic Regression",
                                                            "Random Forest"))
    
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernel coefficient)", ("scale", "auto"), key="gamma")
    
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",
                                                                    "ROC Curve",
                                                                    "Precision-Recall Curve"))
        if st.sidebar.button("Run Model", key="run_model"):
            st.subheader("Support Vector Machine (SVM) Results")
            svc = SVC(C=C,
                    kernel=kernel,
                    gamma=gamma)
            calc_and_plot_metrics(svc, X_train, X_test, y_train, y_test)
    
    
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_lr")
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key="max_iter")

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",
                                                                    "ROC Curve",
                                                                    "Precision-Recall Curve"))
        if st.sidebar.button("Run Model", key="run_model"):
            st.subheader("Logistic Regression Results")
            log_reg = LogisticRegression(C=C, max_iter=max_iter)
            calc_and_plot_metrics(log_reg, X_train, X_test, y_train, y_test)
    
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of Trees", 100, 5000, step=10, key="n_estimators")
        max_depth = st.sidebar.number_input("Max Depth of the Tree", 1, 10, step=1, key="max_depth")
        bootstrap = st.sidebar.radio("Bootstrap Samples", ("True", "False"), key="bootstrap")
        

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",
                                                                    "ROC Curve",
                                                                    "Precision-Recall Curve"))
        if st.sidebar.button("Run Model", key="run_model"):
            st.subheader("Random Forest Results")
            rf = RandomForestClassifier(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        bootstrap=bootstrap,
                                        n_jobs=-1)
            calc_and_plot_metrics(rf, X_train, X_test, y_train, y_test):
    
    
    
    
    
    
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Dataset")
        st.write(df)



if __name__ == "__main__":
    main()