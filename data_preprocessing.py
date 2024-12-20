import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(data_path):
  data = pd.read_csv(data_path)

  categorical_features = ['categorical_feature1', 'categorical_feature2']
  numerical_features = ['numerical_feature1', 'numerical_feature2']

  preprocessor = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), numerical_features),
          ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
      ])

  X = data.drop('target_column', axis=1)
  y = data['target_column']

  X_transformed = preprocessor.fit_transform(X)

  return X_transformed, y