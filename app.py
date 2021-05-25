from flask import Flask, request,render_template, jsonify
import numpy as np
import pickle
import pandas as pd

app=Flask(__name__)
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict_note_authentication():
    '''
    For rendering results on HTML GUI
    '''
    Variance = request.form["Variance"]
    Skewness = request.form["Skewness"]
    Curtosis = request.form["Curtosis"]
    Entropy = request.form["Entropy"]
    final_features = np.array([Variance,Skewness,Curtosis,Entropy]).reshape(1,-1)
    prediction = classifier.predict(final_features)
    print(prediction)
    if prediction == 0:
        output = "not Authentic"
    else:
        output = "Authentic"
        
    return render_template('index.html', prediction_text='The note of this Properties is {}'.format(output))



if __name__=='__main__':
    app.run(debug=True)
    
    