from flask import Flask
from flask import request
from flask import jsonify
from flask import abort
import pickle
import traceback

app = Flask(__name__)

def predict_input(input):
    try:
        input = [input]
        infile = open("./svc_pipeline.pickle",'rb')
        lr = pickle.load(infile)

        label_encodings = {'Bank account or service': 6,
        'Consumer Loan': 1,
        'Credit card': 3,
        'Credit reporting': 4,
        'Debt collection': 0,
        'Money transfers': 8,
        'Mortgage': 2,
        'Other financial service': 9,
        'Payday loan': 7,
        'Prepaid card': 10,
        'Student loan': 5}

        rev_enc = {v: k for k, v in label_encodings.items()}

        index = lr.predict(input)

        return rev_enc[index[0]]
    except:
        return None

@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"

@app.route('/predict', methods=['POST'])
def predict():
    if "input" in request.json:
        inputString = request.json["input"]
        print("inputString: {}".format(inputString))
        prediction = predict_input(inputString)
        if prediction:
            return jsonify({'prediction': prediction})
        else:
            return abort(500)
    else:
        return abort(400)


if __name__ == '__main__':
    app.run(debug=True)