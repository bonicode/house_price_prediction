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
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from os.path import exists

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Home Price Predictor',
                   initial_sidebar_state='auto')


boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

st.write("""
# Home Price Predictor
This app predicts the **Boston Home Price**!
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
else:
    # Train Regression model
    model = RandomForestRegressor()
    model.fit(X, Y)
    # Save SKLEARN model
    joblib.dump(model, 'sk_pipeline.pkl')


# TENSORFLOW
pipeline = None
def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, activation='relu',
            kernel_initializer='normal'))
    model.add(Dense(6, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

if exists('tf_pipeline.pkl'):
    # Load pipeline
    pipeline = joblib.load('tf_pipeline.pkl')
    # Load TF model
    pipeline.named_steps['mlp'].model = keras.models.load_model('keras_model.h5')
else:
    # Train Tensorflow model
    dataframe = pd.read_csv("housing.csv", delim_whitespace=True, header=None)
    dataset = dataframe.values
    X = dataset[:, 0:13]
    Y = dataset[:, 13]

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(
        build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    pipeline.fit(X, Y)
    # Save TF model
    pipeline.named_steps['mlp'].model.save('keras_model.h5')
    # Save pipeline
    pipeline.named_steps['mlp'].model = None
    joblib.dump(pipeline, 'tf_pipeline.pkl')


prediction_tf = pipeline.predict(df)
prediction_sklearn = model.predict(df)

st.header("Predicted Values")
st.write(f'Predicted value with Sklearn: ${prediction_sklearn[0] * 1000:,.2f}')
st.write(f"Predicted value with Tensorflow: ${prediction_tf * 1000:,.2f}")
st.write('---')


# sklearn_mae = np.mean(np.abs(model.predict(X) - Y))
# tf_mae = np.mean(np.abs(pipeline.predict(X) - Y))
sklearn_mae = 0.7788300395256911
tf_mae = 2.611664370020388

st.header("Performance Evaluation")
st.write(f'MAE of Sklearn: ${sklearn_mae * 1000:,.2f}')
st.write(f"MAE of Tensorflow: {tf_mae * 1000:,.2f}")
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')