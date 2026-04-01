import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("🌺 Flower Predictor")
"Enter measurements to predict flower type!"

iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

petal = st.slider("Petal Length", 1.0, 7.0, 4.0)
sepal = st.slider("Sepal Width", 2.0, 4.0, 3.0)

if st.button("🔮 Predict"):
    pred = model.predict([[petal, sepal, 5.0, 3.0]])[0]
    names = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"**{names[pred]}** 🌸")
