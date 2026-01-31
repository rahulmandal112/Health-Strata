import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np

app=Flask(__name__)

# Load the model
model=pickle.load(open('health_dt_model.pkl','rb'))

BOOLEAN_FIELDS = [
    "Diet_Type__Vegan",
    "Diet_Type__Vegetarian",
    "Blood_Group_AB",
    "Blood_Group_B",
    "Blood_Group_O"
]


def normalize_bool(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return 1 if value.lower() == "true" else 0
    return 0

def clean_input(data: dict):
    cleaned = {}

    for key, value in data.items():
        if key in BOOLEAN_FIELDS:
            cleaned[key] = normalize_bool(value)
        else:
            cleaned[key] = value

    return cleaned

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def pedict_api():
    data=request.json['data']
    print(data)
    #print(np.array(list(data.values())).reshape(1,-1))
    #new_data=np.array(list(data.values())).reshape(1,-1)
    cleaned_data = clean_input(data)
    new_data=pd.DataFrame([cleaned_data])
    output=model.predict(new_data)
    print(output[0])
    return jsonify({
        "prediction":int(output[0])})

if __name__ == "__main__":
    app.run(debug=True)
