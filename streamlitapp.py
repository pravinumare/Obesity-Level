import pickle
import pandas as pd
from utils import Gender_encoding, Family_history_with_overweight_encoding, FAVC_encoding, CAEC_encoding, SMOKE_encoding, SCC_encoding, CALC_encoding, MTRANS_encoding, Age_Transformation, NCP_Transformation, converter
import streamlit as st


rf_model_hyp = pickle.load(open('rf_model_hyp.pkl', 'rb'))
columns_list = pickle.load(open('columns_list.obj', 'rb'))


# Creating the title for the app
st.set_page_config(page_title="Obesity Level Prediction",
                   page_icon='💊',
                   layout='centered')

st.header("Obesity Level Predictor 🩺")

Gender = st.selectbox("Select your Gender", ('Male', 'Female'))
Age = st.number_input("Enter your Age")
Height = st.number_input("Enter your Height in meters")
Weight = st.number_input("Enter your Weight in lbs")
family_history_with_overweight = st.selectbox("Do you have family history with overweight?", ('yes', 'no'))
FAVC = st.selectbox("Do you eat high caloric food frequently?", ('yes', 'no'))
FCVC = st.selectbox("Do you usually eat vegetables in your meals?", ('1','2','3'))
NCP = st.selectbox("How many main meals do you have daily?", ('1','2','3','4','5'))
CAEC = st.selectbox("Do you eat any food between meals?", ('Sometimes', 'Frequently', 'Always', 'no'))
SMOKE = st.selectbox("Do you smoke?",('yes', 'no'))
CH2O = st.number_input("How much water do you drink daily? (in ltrs)")
SCC = st.selectbox("Do you monitor the calories you eat daily?", ('yes', 'no'))
FAF = st.selectbox("How often do you have physical activity?", ('0','1', '2', '3'))
TUE = st.selectbox("How much time do you use technological devices such as cell phone, videogames, television, computer and others? (Here 0=not much, 1=sometimes, 2=too much)", ('0', '1', '2'))
CALC = st.selectbox("How often do you drink alcohol?", ('Sometimes', 'Frequently', 'Always', 'no'))
MTRANS = st.selectbox("Which transportation do you usually use?", ('Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'))

button = st.button("Predict")

if button:
    with st.spinner("Loading please wait..."):

        gender = Gender_encoding(Gender)
        age = Age_Transformation(int(Age))
        Height = float(Height)
        Weight = float(Weight)
        fam_history = Family_history_with_overweight_encoding(family_history_with_overweight)
        favc = FAVC_encoding(FAVC)
        fcvc = float(FCVC)
        ncp = NCP_Transformation(float(NCP))
        caec = CAEC_encoding(CAEC)
        smoke = SMOKE_encoding(SMOKE)
        ch2o = float(CH2O)
        ssc = SCC_encoding(SCC)
        faf = float(FAF)
        tue = int(TUE)
        calc = CALC_encoding(CALC)
        trans = MTRANS_encoding(MTRANS)

        test_df = pd.DataFrame({'Gender':[gender], 'Age':[age], 'Height':[Height], 'Weight':[Weight],
                        'family_history_with_overweight':[fam_history], 'FAVC':[favc],
                        'FCVC':[fcvc], 'NCP':[ncp], 'CAEC':[caec], 'SMOKE':[smoke], 'CH2O':[ch2o],
                        'SCC':[ssc], 'FAF':[faf], 'TUE':[tue], 'CALC':[calc], 'MTRANS':[trans]})


        pred = rf_model_hyp.predict(test_df)
        prediction = converter(pred)
        print(prediction)

        st.write(prediction)



