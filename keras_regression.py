import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

# Loading of data
filepath = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
data = pd.read_csv(filepath)

# Separate predictors and target
predictors = data.drop('Strength', axis=1)
target = data['Strength']

# Normalize predictors
predictors_norm = (predictors - predictors.mean()) / predictors.std()

# Number of features
n_cols = predictors.shape[1]

# model
model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(n_cols,)))
model.add(Dense(500, activation='relu'))
model.add(Dense(1))  # regression , hence no activation

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

'''Controls how much output you see during training:
verbose=0 → no output
verbose=1 → progress bar per epoch
verbose=2 → one line per epoch (cleaner in notebooks)'''


if __name__ == "__main__":
    # Train model
    model.fit(predictors_norm, target, validation_split=0.1, epochs=100, verbose=2)
    
    # Test prediction
    preds = model.predict(predictors_norm[:5])
    print("Sample Predictions:", preds.flatten())
