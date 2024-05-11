import io 
import numpy as np
import pandas as pd
import streamlit as st
from utils import load_object,read_yaml

def main():
    st.title("House Price Predictor")
    st.divider()

    uploaded_file = st.file_uploader("Upload a file", type=["csv"])
    schema=read_yaml("config/config.yaml")
    features=schema["year_features"]+schema["numerical_features"]+schema["categorical_features"]
    validation_status=True

    if uploaded_file is not None:

        df=pd.read_csv(uploaded_file)
        df_new=df.drop("Id",axis=1)

        if len(df_new.columns)!=len(features):
            validation_status=False

        for column in df_new.columns:
            if column not in features:
                validation_status=False
                st.write(f"{column} not present in data extracted")

        if validation_status:
            model=load_object("model/model.pkl")
            predictions=model.predict(df)
            df["SalePrice"]=np.expm1(predictions)
            st.write(df)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)  
            csv_content = csv_buffer.getvalue()
            st.download_button(label="Download Data (CSV)",data=csv_content,file_name="result.csv",mime="text/csv")

if __name__ == "__main__":
    main()
