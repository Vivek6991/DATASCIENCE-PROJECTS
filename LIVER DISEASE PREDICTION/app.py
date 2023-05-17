import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Set up the app title
st.title("Liver Disease Prediction")




# Load the Iris dataset
data = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\project-data.csv", sep=';')
data['sex'].replace(['m', 'f'], [0, 1],inplace=True)

#Missing values
data['alkaline_phosphatase'] = data['alkaline_phosphatase'].fillna(data['alkaline_phosphatase'].median())
data['cholesterol'] = data['cholesterol'].fillna(data['cholesterol'].median())
data['albumin'] = data['albumin'].fillna(0)
data['alanine_aminotransferase'] = data['alanine_aminotransferase'].fillna(0)

X=data.iloc[:, 1:12]
y=data.iloc[:, 0]

# Create the models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "SVM": SVC(),
    "Neural Network": MLPClassifier()
}

# Create a function to train and evaluate a model
def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

# Train and evaluate each model
results = {}
for name, model in models.items():
    train_score, test_score = train_model(model, X_train, y_train, X_test, y_test)
    results[name] = {"Train Score": train_score, "Test Score": test_score}


# Allow the user to select a model and see its predictions
selected_model = st.selectbox("Select a model", list(models.keys()))
selected_model = models[selected_model]

# Collect user input
age = st.number_input("Age (years)", min_value=1, max_value=120, step=1)
sex = st.checkbox("Gender: Male")
sex = st.checkbox("Gender: Female")
albumin = st.number_input("Enter albumin")
alkaline_phosphatase = st.number_input("Enter alkaline_phosphatase")
alanine_aminotransferase = st.number_input("Enter alanine_aminotransferase")
aspartate_aminotransferase = st.number_input("Enter aspartate_aminotransferase")
bilirubin = st.number_input("Enter bilirubin")
cholinesterase = st.number_input("Enter cholinesterase")
cholesterol = st.number_input("Enter cholesterol")
creatinina = st.number_input("Enter creatinina")
gamma_glutamyl_transferase = st.number_input("Enter gamma_glutamyl_transferase")

# Make a prediction
predictions = {}
for name, model in models.items():

    input_data = [[albumin, alkaline_phosphatase, alanine_aminotransferase, aspartate_aminotransferase, bilirubin, cholinesterase, cholesterol, creatinina, gamma_glutamyl_transferase, age, sex]]

prediction = model.predict(input_data)[0]
predictions[name] = prediction

# Display the prediction
st.subheader("Predictions")
for name, prediction in predictions.items():
    if prediction == 0:
        st.write(name + ": No Disease")
    elif prediction == 1:
        st.write(name + ": Suspect Disease")
    elif prediction == 2:
        st.write(name + ": Hepatitis")
    elif prediction == 3:
        st.write(name + ": Fibrosis")
    elif prediction == 4:
        st.write(name + ": Cirrhosis")

#Display results of eachmodel
st.subheader("Dataset")
st.write(data)



