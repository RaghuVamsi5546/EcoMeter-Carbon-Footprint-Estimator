import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a ColumnTransformer pipeline for preprocessing.
        """
        try:
            logging.info("Creating preprocessing pipeline...")

            # Numeric columns to scale
            numeric_cols = [
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

            # Categorical columns to encode
            categorical_cols = [
                "heating_type",
                "diet_type"
            ]

            # Numeric pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numeric_cols),
                ("cat", cat_pipeline, categorical_cols)
            ])

            logging.info("Preprocessing pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error("Error creating preprocessing pipeline", exc_info=True)
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies transformations to train and test datasets.
        """
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test datasets loaded for transformation.")

            # Separate input features and target
            target_column = "carbon_footprint"

            X_train = train_df.drop(columns=[target_column, "ID"])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column, "ID"])
            y_test = test_df[target_column]

            # Numeric columns list (same as in get_data_transformer_object)
            numeric_cols = [
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

            # Coerce numeric columns to numeric, invalid entries become NaN
            for col in numeric_cols:
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

            # Get preprocessor
            preprocessor = self.get_data_transformer_object()

            # Fit and transform train, transform test
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data transformation completed successfully.")

            # Save the preprocessor object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)
            logging.info(f"Preprocessor object saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Error during data transformation", exc_info=True)
            raise CustomException(e, sys)

