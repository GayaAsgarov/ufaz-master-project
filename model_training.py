from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  model = DecisionTreeClassifier()
  model.fit(X_train, y_train)

  return model

def evaluate_model(model, X_test, y_test):
  y_pred = model.predict(X_test)

  print("Accuracy:", accuracy_score(y_test, y_pred))
  print(classification_report(y_test, y_pred))