from data_preprocessing import preprocess_data
from model_training import train_model, evaluate_model
from model_explanation import explain_model

X, y = preprocess_data('data.csv')

model = train_model(X, y)

evaluate_model(model, X_test, y_test)

explain_model(model, X_test)