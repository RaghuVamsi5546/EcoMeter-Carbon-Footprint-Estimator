import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    """
    Configuration class that holds file paths for data ingestion outputs.
    """
    raw_data_path: str = os.path.join("notebook", "data", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    """
    Class to handle reading raw data and splitting it into training and testing sets.
    """

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """
        Reads the raw dataset, splits it into training and testing datasets, and saves them.

        Returns:
            Tuple[str, str]: (train_data_path, test_data_path)
        """
        logging.info("Entered the data ingestion method/component")

        try:
            if not os.path.exists(self.ingestion_config.raw_data_path):
                logging.error(f"Raw data file not found at {self.ingestion_config.raw_data_path}")
                raise FileNotFoundError(f"File not found: {self.ingestion_config.raw_data_path}")

            # Read raw dataset
            df = pd.read_csv(self.ingestion_config.raw_data_path, encoding="utf-8")
            logging.info("Dataset loaded successfully")

            # Ensure artifact directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Split the data
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            logging.info(f"Train shape: {train_set.shape}")
            logging.info(f"Test shape: {test_set.shape}")

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(
                f"Train and test data saved to {self.ingestion_config.train_data_path} and {self.ingestion_config.test_data_path}"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
