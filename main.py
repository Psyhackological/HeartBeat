"""Final project: Heart Disease Prediction"""

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.metrics import classification_report  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


class BaseModel:
    """Base class for models"""

    def train(self, data: pd.DataFrame) -> None:
        """Train the model with data"""
        raise NotImplementedError

    def predict(self, data: pd.DataFrame) -> Any:
        """Predict using the trained model with data"""
        raise NotImplementedError


class HeartDiseasePredictor(BaseModel):
    """Heart Disease Predictor class"""

    def __init__(self) -> None:
        self.models = [
            RandomForestClassifier() for _ in range(3)
        ]  # TODO: Implement more models

    def train(self, data: pd.DataFrame) -> None:
        """Train the HeartDiseasePredictor models with data"""
        x_train_data = data.drop("target", axis=1)
        y_train_data = data["target"]
        for model in self.models:
            model.fit(x_train_data, y_train_data)

    def predict(self, data: pd.DataFrame) -> Any:
        """Predict using the trained HeartDiseasePredictor models with data"""
        predictions = [model.predict(data) for model in self.models]
        return predictions


def log_results(func: Any) -> Any:
    """Decorator for logging function results"""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} returned {result:.2f}")
        return result

    return wrapper


@log_results
def calculate_average_age(data: pd.DataFrame) -> float:
    """Calculate the average age of patients"""
    return data["age"].mean()


df = pd.read_csv("heart.csv")
x = df.drop("target", axis=1)
y = df["target"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

predictor = HeartDiseasePredictor()
predictor.train(pd.concat([x_train, y_train], axis=1))
model_predictions = predictor.predict(x_test)  # list of predictions for each model

# Loop to print out the accuracy of each model
for i, pred in enumerate(model_predictions):
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)
    print(f"Model {i+1} Accuracy: {accuracy:.2f}")
    print(f"Model {i+1} Classification Report:\n{report}")

average_age = calculate_average_age(df)

print(f"Average age: {average_age:.2f}")
