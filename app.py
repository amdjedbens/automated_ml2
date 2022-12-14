import os

import pandas as pd
import pandas_profiling
import streamlit as st
#ML stuff
from pycaret.classification import compare_models, pull, save_model, setup
from pycaret.regression import compare_models, pull, save_model, setup
from streamlit_pandas_profiling import st_profile_report

with st.sidebar:
    st.image("https://cdn.pixabay.com/photo/2018/09/18/11/19/artificial-intelligence-3685928_1280.png")
    st.title("EasyAutoML")
    choice = st.radio("Navigation",["Data loading","Exploratory","Modeling","Download"])
    st.info("This application to explore your data & build an automated ML pipeline.")


if os.path.exists("source_data.csv"):
    df = pd.read_csv("source_data.csv", index_col=None)


if choice == "Data loading":
    st.title("Upload your data for modeling") 
    file = st.file_uploader("Upload your dataset here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("source_data.csv", index=None)
        st.dataframe(df)

elif choice == "Exploratory":
    st.title('Automated EDA')
    profile_report = df.profile_report()
    st_profile_report(profile_report)
    
elif choice == "Modeling":
    
    st.title('Time for ML')
    target = st.selectbox('Choose the target column', df.columns)
    if st.button("Train model"):
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info("This is the ML experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')
elif choice == "Download":
    with open("best_model.pkl",'rb') as f:
        st.download_button("Download the model file",f,"best_model.pkl")
else:
    pass

st.write("Made with <3 by Amdjed")
