import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import joblib


__version__ = "1.0.0"


class Prediction:
    def __init__(self, model_name):
        scaler_path = Path(__file__).parent / "std_scaler.pkl"
        feature_path = Path(__file__).parent / "feature_columns.pkl"
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(feature_path)

        model_path = Path(__file__).parent / model_name
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]["index"]
        self.output_details = self.interpreter.get_output_details()[0]["index"]

    def make_prediction(self, weight, height, gender):
        # Convert gender to numeric format
        gender_value = 1 if gender.lower() == "male" else 0

        # Set input tensor
        input_data = pd.DataFrame(
            [[gender_value, height, weight]], columns=self.feature_columns
        ).astype(np.float32)
        input_data = self.scaler.transform(input_data)
        self.interpreter.set_tensor(self.input_details, input_data)

        # Run inference
        self.interpreter.invoke()

        # Get prediction
        yhat = self.interpreter.get_tensor(self.output_details)
        prediction = np.argmax(yhat, axis=1)[0]
        return prediction

    def __str__(self):
        return f"Model Path: {self.model}"
