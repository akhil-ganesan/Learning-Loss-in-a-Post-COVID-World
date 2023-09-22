# -*- coding: utf-8 -*-
#!/usr/bin/env tf
"""
Created on Tue Apr 25 14:57:07 2023

@author: akhil
"""
# Libraries
import csv
import numpy as np
from tensorflow import keras

# Dataset & Preprocessing
full_loss = [] # Total Learning Loss By State
full_indicators = [] # Relevant Data Point About the State
data = open("LearningLoss.csv")

trainingCuttoff = 35

csv_data = csv.reader(data)

i = 0

for row in csv_data:
    if i != 0:
        full_loss.append(np.asarray(float(row[5])))
        full_indicators.append(np.asarray(row[1:5]).astype(float))
    i += 1

full_loss = np.reshape(full_loss, (-1, 1))
full_indicators = np.reshape(full_indicators, (51, 4))
# This standardizes the data for analysis
full_indicators = np.divide(full_indicators, full_indicators.max(axis=0, keepdims=True))
maxLoss = full_loss.max(axis=0, keepdims=True)
full_loss = np.divide(full_loss, full_loss.max(axis=0, keepdims=True))

train_x = full_indicators[:trainingCuttoff]
test_x = full_indicators[trainingCuttoff:]
train_y = full_loss[:trainingCuttoff]
test_y = full_loss[trainingCuttoff:]

print(full_loss.shape)
print(full_indicators.shape)

#Linear Model/Learning
model = keras.Sequential([
    keras.layers.Dense(5, input_shape=(4,), activation = 'tanh'),
    keras.layers.Dense(3, activation = 'softmax'),
    keras.layers.Dense(1)#, activation = 'sigmoid')
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy']
    )

model.fit(train_x, train_y, epochs=1000)

train_loss, train_acc = model.evaluate(train_x, train_y)
test_loss, test_acc = model.evaluate(test_x, test_y)
print('Loss on training data: {}%'.format(train_loss))
print('Loss on testing data: {}%'.format(test_loss))

def mean_absolute_error(y, pred):
    a = 0
    b = 0
    for i in keras.losses.mean_absolute_error(y, pred):
        a += i
        b += 1
    return float(a)/b

#graph(np.reshape(range(0, 51), (-1, 1)), full_loss)
print(mean_absolute_error(test_y, model.predict(test_x))*maxLoss[0][0])

def savePredictions():
    csv_wr = csv.writer(open('EstimatedLoss.csv', 'w'), delimiter=',')
    for i in np.reshape(model.predict(full_indicators), (-1)):
        csv_wr.writerow([i*maxLoss[0][0]])

savePredictions()
