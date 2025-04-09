import logging
from utils.data_loader import load_data, preprocess_data, split_data
from utils.model import train_model, save_model
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

df = load_data("data/Admission.xlsx")
X_scaled, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X_scaled, y)

model = train_model(X_train, y_train)
save_model(model, "models/model.pkl")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)

logging.info(f"Model accuracy on test data: {acc:.2%}")
logging.info(f"Confusion matrix:\n{conf}")