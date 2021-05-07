from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('RandomForest.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("cropified.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=np.array([[float(x) for x in request.form.values()]])
    print(int_features)
    output=model.predict(int_features)
    final_output=output[0]
    
    return render_template('cropified.html',pred='You should grow {} in your land'.format(final_output))
    

if __name__ == '__main__':
    app.run(debug=True)
