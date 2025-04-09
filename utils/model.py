from sklearn.neural_network import MLPClassifier
import joblib
import logging

def train_model(X_train, y_train):
    logging.info("Training MLPClassifier model.")
    model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=1)
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    return model

def save_model(model, path="model.pkl"):
    joblib.dump(model, path)
    logging.info(f"Model saved to {path}.")

def load_model(path="model.pkl"):
    return joblib.load(path)