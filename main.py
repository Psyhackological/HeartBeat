import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart.csv")
x = df.drop("target", axis=1)
y = df["target"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


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
        self.model = RandomForestClassifier()

    def train(self, data: pd.DataFrame) -> None:
        """Train the HeartDiseasePredictor model with data"""
        x_train_data = data.drop("target", axis=1)
        y_train_data = data["target"]
        self.model.fit(x_train_data, y_train_data)

    def predict(self, data: pd.DataFrame) -> Any:
        """Predict using the trained HeartDiseasePredictor model with data"""
        return self.model.predict(data)
