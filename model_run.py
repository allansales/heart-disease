import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#import sys
#data_path = sys.argv[1]
#model_path = sys.argv[2]
#output_path = sys.argv[2]

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

# Data preprocessing
encoder = LabelEncoder()
new_data['famhist']=encoder.fit_transform(new_data['famhist'])
scale = MinMaxScaler(feature_range =(0,100))
new_data['sbp'] = scale.fit_transform(new_data['sbp'].values.reshape(-1,1))

# Load model
clf = load(args.model) 

# Predict heart disease
pred = clf.predict(new_data)

# Add prediction to raw data
new_data = new_data.assign(chd=pred)

# Write results
new_data.to_csv(args.output, index=False)