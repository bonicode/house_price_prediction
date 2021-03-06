from os.path import exists

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import keras

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Home Price Predictor',
                   initial_sidebar_state='auto')


boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

st.write("""
# Boston Home Price Predictor
This app predicts the **Boston Home Price** with some regression models implemented in Sklearn and Tensorflow.

GitHub Repository: https://github.com/BoniOloff/house_price_prediction
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')
def user_input_features():
    CRIM = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    ZN = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())
    INDUS = st.sidebar.slider('INDUS', X.INDUS.min(),
                              X.INDUS.max(), X.INDUS.mean())
    CHAS = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
    NOX = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())
    RM = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())
    AGE = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())
    DIS = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())
    RAD = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.mean())
    TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())
    PTRATIO = st.sidebar.slider(
        'PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
    B = st.sidebar.slider('B', X.B.min(), X.B.max(), X.B.mean())
    LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min(),
                              X.LSTAT.max(), X.LSTAT.mean())
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('User inputs:')
params1 = df.iloc[:, :6]
params2 = df.iloc[:, 6:]
st.write(params1, params2)
st.write('---')

# SKLEARN
model = None
if exists('sk_pipeline.pkl'):
    # Load SKLEARN model
    model = joblib.load('sk_pipeline.pkl')

# TENSORFLOW
pipeline = None
if (exists('tf_pipeline.pkl') and exists('keras_model.h5')):
    # Load pipeline
    pipeline = joblib.load('tf_pipeline.pkl')
    # Load TF model
    pipeline.named_steps['mlp'].model = keras.models.load_model('keras_model.h5')

prediction_tf = pipeline.predict(df)
prediction_sklearn = model.predict(df)

st.header("Predicted Values")
st.write(f'Predicted value with RandomForestRegressor in Sklearn: ${prediction_sklearn[0] * 1000:,.2f}')
st.write(f"Predicted value with deeplearning regression in Tensorflow: ${prediction_tf * 1000:,.2f}")
st.write('---')


# sklearn_mae = np.mean(np.abs(model.predict(X) - Y))
# tf_mae = np.mean(np.abs(pipeline.predict(X) - Y))
sklearn_mae = 0.7788300395256911
tf_mae = 2.611664370020388

st.header('Performance Evaluation')
st.write('From this model we can see that Random Forest perform better than DeepLearning.')
st.write(f'- MAE of RandomForestRegressor in Sklearn: ${sklearn_mae * 1000:,.2f}')
st.write(f'- MAE of Tensorflow: {tf_mae * 1000:,.2f}')
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

st.header('Feature Importance')
if exists("gg1.png"):
    st.image("gg1.png")
else:
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    plt.savefig("gg1.png",dpi=150, bbox_inches='tight')
    st.pyplot(bbox_inches='tight')
st.write('---')

if exists("gg2.png"):
    st.image("gg2.png")
else:
    plt.title('Feature importance based on SHAP values (Bar)')
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.savefig("gg2.png",dpi=150, bbox_inches='tight')
    st.pyplot(bbox_inches='tight')
st.write('---')


st.header("References:")
st.write("""
- https://github.com/slundberg/shap
- https://www.youtube.com/watch?v=JwSS70SZdyM&t=97s
- https://github.com/DavidCico/Boston-House-Prices-With-Regression-Machine-Learning-and-Keras-Deep-Learning
- https://stackoverflow.com/questions/37984304/how-to-save-a-scikit-learn-pipline-with-keras-regressor-inside-to-disk
""")
