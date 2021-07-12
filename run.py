import pickle
import sklearn

def predict(input):
    infile = open("./svc_pipeline.pickle",'rb')
    lr = pickle.load(infile)

    if not lr:
        return None

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

input = ["There was a problem in my bank account"]
print(predict(input))

