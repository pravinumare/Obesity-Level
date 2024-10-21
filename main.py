from flask import Flask,request,jsonify
import pickle
import pandas as pd
from utils import Gender_encoding, Family_history_with_overweight_encoding, FAVC_encoding, CAEC_encoding, SMOKE_encoding, SCC_encoding, CALC_encoding, MTRANS_encoding, Age_Transformation, NCP_Transformation, converter  

app = Flask(__name__)


rf_model_hyp = pickle.load(open('rf_model_hyp.pkl', 'rb'))
columns_list = pickle.load(open('columns_list.obj', 'rb'))


@app.route('/ObesityLevelPrediction')
def ObesityLevelPrediction():

    data = request.get_json()

    Gender = data['Gender']
    Age = data['Age']
    Height = data['Height']
    Weight = data['Weight']
    family_history_with_overweight = data['family_history_with_overweight']
    FAVC = data['FAVC'] 
    FCVC = data['FCVC'] 
    NCP = data['NCP'] 
    CAEC = data['CAEC'] 
    SMOKE = data['SMOKE'] 
    CH2O = data['CH2O'] 
    SCC = data['SCC'] 
    FAF = data['FAF'] 
    TUE = data['TUE'] 
    CALC = data['CALC'] 
    MTRANS = data['MTRANS'] 

    gender = Gender_encoding(Gender)
    age = Age_Transformation(Age)
    family_history_with_overweight = Family_history_with_overweight_encoding(family_history_with_overweight)
    favc = FAVC_encoding(FAVC)
    fcvc = FCVC
    ncp = NCP_Transformation(NCP)
    caec = CAEC_encoding(CAEC)
    smoke = SMOKE_encoding(SMOKE)
    ch2o = CH2O
    scc = SCC_encoding(SCC)
    faf = FAF
    tue = TUE
    calc = CALC_encoding(CALC)
    trans = MTRANS_encoding(MTRANS)

    test_df = pd.DataFrame({'Gender':[gender], 'Age':[age], 'Height':[Height], 'Weight':[Weight],
                        'family_history_with_overweight':[family_history_with_overweight], 'FAVC':[favc],
                        'FCVC':[fcvc], 'NCP':[ncp], 'CAEC':[caec], 'SMOKE':[smoke], 'CH2O':[ch2o],
                        'SCC':[scc], 'FAF':[faf], 'TUE':[tue], 'CALC':[calc], 'MTRANS':[trans]})
    
    pred = rf_model_hyp.predict(test_df)

    prediction  = converter(pred[0])

    return jsonify({'Prediction' : prediction})


if __name__ == '__main__':
    app.run()