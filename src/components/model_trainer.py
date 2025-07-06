import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    target_column: str = "carbon_footprint"


class ModelTrainer:
    """
    Handles model training and selection.
    """

    NUMERIC_COLUMNS = [
        "electricity_kwh_per_month",
        "natural_gas_therms_per_month",
        "vehicle_miles_per_month",
        "house_area_sqft",
        "water_usage_liters_per_day",
        "public_transport_usage_per_week",
        "household_size",
        "home_insulation_quality",
        "meat_consumption_kg_per_week",
        "laundry_loads_per_week",
        "recycles_regularly",
        "composts_organic_waste",
        "uses_solar_panels",
        "energy_efficient_appliances",
        "owns_pet",
        "smart_thermostat_installed"
    ]

    def __init__(self):
        self.config = ModelTrainerConfig()

    def evaluate_models(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series,
        models: dict,
        param_grids: dict
    ) -> tuple[dict, dict]:
        """
        Train and evaluate multiple models with GridSearchCV.

        Returns:
            Tuple:
                - dict: model_name -> R2 score
                - dict: model_name -> best_estimator_
        """
        report = {}
        best_models = {}

        for name, model in models.items():
            logging.info(f"Training {name} with hyperparameter tuning...")

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=3,
                n_jobs=-1,
                verbose=1,
                scoring="r2"
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            preds = best_model.predict(X_test)
            score = r2_score(y_test, preds)

            report[name] = score
            best_models[name] = best_model

            logging.info(f"{name} best params: {grid_search.best_params_}")
            logging.info(f"{name} R2 Score: {score:.4f}")

        return report, best_models

    def initiate_model_trainer(
        self,
        train_path: str,
        test_path: str,
        preprocessor_path: str
    ) -> tuple[str, float, str]:
        """
        Orchestrates model training and selection.

        Returns:
            Tuple containing:
                - Best model name
                - R2 score
                - Saved model path
        """
        try:
            logging.info("Starting model training pipeline...")

            train_df = pd.read_csv(train_path, encoding="utf-8")
            test_df = pd.read_csv(test_path, encoding="utf-8")

            logging.info("Train and test data loaded successfully.")

            target_column = self.config.target_column
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Loading preprocessor object...")
            preprocessor = load_object(preprocessor_path)

            # Ensure numeric columns are coerced to numeric
            for col in self.NUMERIC_COLUMNS:
                if col in X_train.columns:
                    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
                    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

            logging.info("Transforming data using preprocessor...")
            X_train = preprocessor.transform(X_train)
            X_test = preprocessor.transform(X_test)

            # Define models
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "SVR": SVR(),
                "CatBoost": CatBoostRegressor(verbose=0),
                "XGBoost": XGBRegressor(objective="reg:squarederror")
            }

            # Define parameter grids
            param_grids = {
                "LinearRegression": {},
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso": {"alpha": [0.001, 0.01, 0.1]},
                "ElasticNet": {
                    "alpha": [0.001, 0.01, 0.1],
                    "l1_ratio": [0.2, 0.5, 0.8]
                },
                "DecisionTree": {
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10]
                },
                "RandomForest": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5, 7]
                },
                "GradientBoosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 1.0]
                },
                "SVR": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"]
                },
                "CatBoost": {
                    "iterations": [50, 100],
                    "learning_rate": [0.05, 0.1]
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                }
            }

            # Evaluate models
            report, best_models = self.evaluate_models(
                X_train, y_train, X_test, y_test, models, param_grids
            )

            # Select the best model
            best_model_name = max(report, key=report.get)
            best_model_score = report[best_model_name]
            best_model = best_models[best_model_name]

            logging.info(f"Best model: {best_model_name} with R2 Score: {best_model_score:.4f}")

            # Save the best model
            save_object(self.config.trained_model_file_path, best_model)
            logging.info(f"Model saved to {self.config.trained_model_file_path}")

            return best_model_name, best_model_score, self.config.trained_model_file_path

        except Exception as e:
            logging.error("Error during model training", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        best_model_name, r2, model_path = trainer.initiate_model_trainer(
            train_path="artifacts/train.csv",
            test_path="artifacts/test.csv",
            preprocessor_path="artifacts/preprocessor.pkl"
        )

        print("Training complete.")
        print(f"Best Model: {best_model_name}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Model saved at: {model_path}")

    except Exception as e:
        print("Error occurred during training.")
        print(e)
