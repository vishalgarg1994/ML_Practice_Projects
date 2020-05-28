from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All3"

@app.route('/predict',methods=["Get"])
def predict_salary():
    
    """Let's get Salary based on experience3
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: experience
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    experience=request.args.get("experience")
    experience = np.array([[int(experience)]])
    prediction=classifier.predict(experience)
    print(prediction)
    return "Hello The answer is"+str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_salary_file():
    """Let's get Salary based on experience2
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    return str(list(prediction))


if __name__=='__main__':
    app.run()