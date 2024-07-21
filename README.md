# About this project

We want to learning how to optimize model linear regression by using "Gradient descent"

Workflow

1. Dataset preprocess

a) Filter columns 
- select only 3 columns -> ["Hours Studied", "Previous Scores", "Sample Question Papers Practiced"] from all columns
- rename selected columns as {"Hours Studied" : "x1", "Sample Question Papers Practiced" : "x2", "Previous Scores" : "y"}

b) Split dataset
- split dataset into 2 parts -> training and testing
- save as training.csv and testing.csv

2. Training 
- read training.csv and convert to dataframe
- convert data in each column to list or array
- train to find optimal weight
- plot scatter y_true
- plot graph y_pred = (w1*x1) + (w2*x2) + b
- return [w1, w2, b]

3. Testing
- get [w1, w2, b] from Training
- calculate y_pred from y_pred = (w1*x1) + (w2*x2) + b
- calculate cost
- plot scatter y_true
- plot graph y_pred = (w1*x1) + (w2*x2) + b






