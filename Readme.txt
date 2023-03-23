In this program, we first import the required libraries, including pandas for data manipulation
numpy for numerical operations, and scikit-learn for machine learning. We then load the dataset
of house prices and split it into input and output variables, where X represents the input 
features and y represents the target variable (i.e., the price). We then split the data into
training and testing sets using the train_test_split function, where 20% of the data is used for testing.

Next, we create a linear regression model using the LinearRegression class and
train it on the training data using the fit method. We then make predictions on the testing 
data using the predict method and evaluate the performance of the model using the mean squared error and root mean squared error.

Note that this is just a sample program and may need to be modified or customized for a
specific problem statement or dataset. Additionally, this program assumes that the dataset
is already preprocessed and does not include any feature engineering or data cleaning steps.