import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# House Price Prediction App

This app tries to predict the price of houses in Boston.
""")

st.write("---")

# Load boston housing dataset.
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
st.sidebar.header("Specify Input Parameters")

def input_user():
    CRIM = st.sidebar.slider("CRIM", X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    ZN   = st.sidebar.slider("ZN", X.ZN.min(), X.ZN.max(), X.ZN.mean())