from sklearn.neural_network import MLPClassifier
import joblib

def train_model(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=500, random_state=1)
    model.fit(X_train, y_train)
    return model

def save_model(model, path="model.pkl"):
    joblib.dump(model, path)

def load_model(path="model.pkl"):
    return joblib.load(path)