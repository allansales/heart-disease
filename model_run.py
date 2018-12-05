import pandas as pd
import math

from joblib import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--input", dest="input",
                    help="Input file path", metavar="INPUT")

parser.add_argument("-o", "--output", dest="output",
                    help="Output file path", metavar="OUTPUT")

parser.add_argument("-m", "--model", dest="model",
                    help="Model file path", metavar="MODEL")

args = parser.parse_args()

# Data load
new_data = pd.read_csv(args.input)

column = ['sbp','tobacco','ldl','adiposity','famhist','type','obesity','alcohol','age']
new_data.columns=column

# Data preprocessing
encoder = LabelEncoder()
new_data['famhist']=encoder.fit_transform(new_data['famhist'])
scale = MinMaxScaler(feature_range =(0,100))
new_data['sbp'] = scale.fit_transform(new_data['sbp'].values.reshape(-1,1))

# Load model or calculate Logistic regression
if(args.model != None):
    clf = load(args.model)

    # Predict heart disease
    pred = clf.predict(new_data)

else:
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

        # return class of instance
    def define_class(x):
        if (x < 0.5):
            return 0
        return 1

    ## Logit coefficients
    logit_coef = [0.00828798, 0.10091803, 0.16580194, 0.04199072, -0.8685583, 0.01429065, -0.1419972, -0.00107974, 0.02339634]
    logit_intercept = -0.78985061

    prob = new_data.multiply(logit_coef, axis=1).sum(axis=1) + logit_intercept
    prob = prob.apply(sigmoid)

    # predict heart disease
    pred = prob.apply(define_class)

# Add prediction to raw data
new_data = new_data.assign(chd=pred)

# Write results
new_data.to_csv(args.output, index=False)
