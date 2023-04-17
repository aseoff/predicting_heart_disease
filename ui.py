import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import random
import matplotlib.pyplot as plt

df = pd.read_csv('heart.csv')

df = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

X_train, X_test, y_train, y_test = train_test_split(df.drop(['HeartDisease'], axis=1), df['HeartDisease'], test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

def predict_and_chart(model, model_name, X_test, y_test):
    # Predict on 10 random samples
    random.seed(42)
    sample_indices = random.sample(range(len(X_test)), 10)
    samples = X_test.iloc[sample_indices]
    sample_preds = model.predict(samples)
    sample_actuals = y_test.iloc[sample_indices].values
    
    # Print sample features
    st.write(f"Sample features for {model_name}:")
    st.write(samples)
    st.write("-------------------------")
    
    # Create chart of predictions for each sample
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"{model_name} Predictions for 10 Random Samples")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Heart Disease (1) or No Heart Disease (0)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No Heart Disease", "Heart Disease"])
    ax.scatter(sample_indices, sample_actuals, label="Actual", s=200)
    ax.scatter(sample_indices, sample_preds, label="Predicted", marker="s", s=200)
    ax.legend()
    
    # Print whether each prediction was correct or incorrect
    for i, (pred, actual) in enumerate(zip(sample_preds, sample_actuals)):
        if pred == actual:
            st.write(f"{model_name} predicted {pred} correctly")
        else:
            st.write(f"{model_name} predicted {pred}, but actual is {actual}")
        st.write("-------------------------")
    st.write("----------------------------")

st.title("Heart Disease Prediction App")

# Display the dataset
st.header("Heart Disease Dataset")
st.write(df)

# Display the accuracy scores
st.header("Model Accuracies")
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
lr_pred = lr.predict(X_test)
knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)


#Evaluate the models and display the accuracy scores
dt_accuracy = accuracy_score(y_test, dt_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)

st.write("Decision Tree Accuracy:", dt_accuracy)
st.write("Random Forest Accuracy:", rf_accuracy)
st.write("Logistic Regression Accuracy:", lr_accuracy)
st.write("KNN Accuracy:", knn_accuracy)
st.write("SVM Accuracy:", svm_accuracy)

#Display the predictions and charts for each model
st.header("Model Predictions")
predict_and_chart(dt, "Decision Tree", X_test, y_test)
predict_and_chart(rf, "Random Forest", X_test, y_test)
predict_and_chart(lr, "Logistic Regression", X_test, y_test)
predict_and_chart(knn, "KNN", X_test, y_test)
predict_and_chart(svm, "SVM", X_test, y_test)