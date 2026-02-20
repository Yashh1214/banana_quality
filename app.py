import streamlit as st

import numpy as np
import pickle
with open("transformer.pkl", "rb") as f:
    transformer=pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
st.title("Banana Quality Score")
variety=st.selectbox('variety',['Plantain', 'Blue Java', 'Red Dacca', 'Burro', 'Cavendish',
       'Manzano', 'Fehi', 'Lady Finger'])

region=st.selectbox('region',['Costa Rica', 'Philippines', 'Honduras', 'India', 'Colombia',
       'Brazil', 'Ecuador', 'Guatemala'])
ripeness=st.number_input('ripeness_index')
sugar_content=st.number_input('sugar_content_brix')
firmness=st.number_input('firmness_kgf')
length=st.number_input("length_cm")
weight=st.number_input("weight_g")
tree_age=st.number_input("tree_age_years")
altitude=st.number_input("altitude_m")
rainfall=st.number_input("rainfall_mm")
soil=st.number_input("soil_nitrogen_ppm")

submit=st.button("Predict Quality Score")
if submit:
    features=np.array([[variety,region,ripeness,sugar_content,firmness,length,weight,tree_age,altitude,rainfall,soil]])
    scaled_features=transformer.transform(features)
    score=model.predict(scaled_features)

    st.write("Banana Quality Score:",score)