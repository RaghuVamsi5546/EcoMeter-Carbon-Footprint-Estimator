import os
import sys
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Handles data preprocessing and transformation.
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

    CATEGORICAL_COLUMNS = [
        "heating_type",
        "diet_type"
    ]

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates and returns a ColumnTransformer pipeline for preprocessing.

        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """
        try:
            logging.info("Creating preprocessing pipeline...")

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, self.NUMERIC_COLUMNS),
                ("cat", cat_pipeline, self.CATEGORICAL_COLUMNS)
            ])

            logging.info("Preprocessing pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error("Error creating preprocessing pipeline", exc_info=True)
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, 
        train_path: str, 
        test_path: str
    ) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, str]:
        """
        Applies transformations to train and test datasets.

        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the test dataset.

        Returns:
            Tuple containing:
                - Transformed training features
                - Transformed testing features
                - Training target
                - Testing target
                - Path to saved preprocessor object
        """
        try:
            train_df = pd.read_csv(train_path, encoding="utf-8")
            test_df = pd.read_csv(test_path, encoding="utf-8")

            logging.info("Train and test datasets loaded for transformation.")

            target_column = "carbon_footprint"

            # Validate required columns exist
            missing_numeric = [col for col in self.NUMERIC_COLUMNS if col not in train_df.columns]
            missing_categorical = [col for col in self.CATEGORICAL_COLUMNS if col not in train_df.columns]
            if missing_numeric or missing_categorical:
                raise CustomException(
                    f"Missing columns in training data. "
                    f"Numeric: {missing_numeric}, Categorical: {missing_categorical}",
                    sys
                )

            # Separate input features and target
            drop_cols = [target_column] + [c for c in ["ID"] if c in train_df.columns]
            X_train = train_df.drop(columns=drop_cols)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=drop_cols)
            y_test = test_df[target_column]

            # Coerce numeric columns to numeric
            for col in self.NUMERIC_COLUMNS:
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

            preprocessor = self.get_data_transformer_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data transformation completed successfully.")

            os.makedirs(
                os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path),
                exist_ok=True
            )
            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info(f"Preprocessor object saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            ) # type: ignore

        except Exception as e:
            logging.error("Error during data transformation", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        train_path = os.path.join("artifacts", "train.csv")
        test_path = os.path.join("artifacts", "test.csv")

        transformer = DataTransformation()
        (
            X_train_transformed,
            X_test_transformed,
            y_train,
            y_test,
            preprocessor_path
        ) = transformer.initiate_data_transformation(train_path, test_path)

        logging.info("Standalone transformation executed successfully.")
        logging.info(f"Transformed train shape: {X_train_transformed.shape}")
        logging.info(f"Transformed test shape: {X_test_transformed.shape}")
        logging.info(f"Preprocessor saved to: {preprocessor_path}")

    except Exception as e:
        logging.error("Error during standalone transformation test", exc_info=True)
