import os
import sys
import pandas as pd
import numpy as np

from src.utils import load_object
from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    """
    Pipeline to load preprocessor and model, and make predictions on new data.
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

    def __init__(
        self,
        preprocessor_path: str = "artifacts/preprocessor.pkl",
        model_path: str = "artifacts/model.pkl"
    ):
        self.preprocessor_path = preprocessor_path
        self.model_path = model_path
        logging.info("PredictPipeline initialized.")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on a DataFrame of new data.

        Args:
            data (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted target values.
        """
        try:
            logging.info("Loading preprocessor and model...")
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            # Coerce numeric columns to numeric
            for col in self.NUMERIC_COLUMNS:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors="coerce")

            logging.info("Transforming input data...")
            transformed_data = preprocessor.transform(data)

            logging.info("Generating predictions...")
            predictions = model.predict(transformed_data)

            return predictions

        except Exception as e:
            logging.error("Error during prediction.", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Example input data with dummy values (replace with real data)
        sample_data = pd.DataFrame({
            "electricity_kwh_per_month": [300],
            "natural_gas_therms_per_month": [20],
            "vehicle_miles_per_month": [500],
            "house_area_sqft": [1500],
            "water_usage_liters_per_day": [500],
            "public_transport_usage_per_week": [3],
            "household_size": [4],
            "home_insulation_quality": [7],
            "meat_consumption_kg_per_week": [2],
            "laundry_loads_per_week": [5],
            "recycles_regularly": [1],
            "composts_organic_waste": [0],
            "uses_solar_panels": [1],
            "energy_efficient_appliances": [1],
            "owns_pet": [0],
            "smart_thermostat_installed": [1],
            "heating_type": ["electric"],
            "diet_type": ["omnivore"]
        })

        pipeline = PredictPipeline()
        predictions = pipeline.predict(sample_data)
        print("Predictions:", predictions)

    except Exception as e:
        print("Error during prediction test:")
        print(e)
