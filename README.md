# UCLA Admission Predictor

This project uses a neural network model to predict student admission to UCLA based on academic scores and research experience.

## Project Structure

- `utils/`: Core logic including data loading, preprocessing, and model definition
- ``: Script to train and evaluate the model
- ``: Streamlit web application interface
- `Admission.xlsx`: Source data file
- `model.pkl`: Trained neural network model
- `scaler.pkl`: Scaler object used to standardize input features
- `requirements.txt`: Python dependencies

## How to Use

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Train the model:
   ```
   python train_and_save_model.py
   ```

3. Launch the web application:
   ```
   streamlit run app.py
   ```

## Model

- Model: MLPClassifier from scikit-learn
- Hidden layers: (16, 8)
- Target: Binary classification (Admitted or Not Admitted)

## Logging

Training and data processing steps are logged for easier debugging.
## Author

- Name: Luyu
- Student number:040986748 
