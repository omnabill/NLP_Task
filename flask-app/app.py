import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('final_prediction.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    final_features = list(request.form.values())

    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='Industry is: '+' {} '.format(output.upper()))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([str(data.values())])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)