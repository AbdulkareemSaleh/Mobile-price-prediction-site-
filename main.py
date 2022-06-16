# import flask
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('clf.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
  battery_power=request.form['battery_power']
  bluetooth=request.form['bluetooth']
  front_camera=request.form['front_camera']
  four_g=request.form['four_g']
  int_memory=request.form['int_memory']
  n_cores=request.form['n_cores']
  pc=request.form['pc']
  touch_screen=request.form['touch_screen']
  wifi=request.form['wifi']
  
  

  
  data=pd.DataFrame({
    'battery_power':battery_power,
    'bluetooth' :bluetooth,
    'front_camera':front_camera,
    'four_g':four_g,
    'int_memory':int_memory,
    'n_cores':n_cores,
    'pc':pc,
    'touch_screen':touch_screen,
    'wifi':wifi
    
    
  },index=[0])
  # Model Prediction
  prediction = model.predict(data)
  
  if prediction==0:
    return render_template('index.html', prediction_text='The mobile cost is low')
  elif prediction==1:
    return render_template('index.html', prediction_text='The mobile cost is medium')
  elif prediction==2:
    return render_template('index.html', prediction_text='The mobile cost is high')
  elif prediction==3:
     return render_template('index.html', prediction_text='The mobile cost is very high')
    
    

    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')