"""
CheXpert Data Preprocessor
- Splitting: Patient-level split (prevents data leakage)
- Cleaning: Handles NaN values
- Output: Creates train.csv, val.csv, and train_tiny.csv (10 patients)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
import logging
import yaml
import sys
from common import setup_logger

logger = setup_logger("CheXpertPreprocessor")

class CheXpertPreprocessor:
    def __init__(self,input_dir:str, output_dir: str, config_path: str = "configs/data_preprocessing.yaml"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)

        self.csv_path = self.input_dir / 'train.csv'

        if not self.csv_path.exists():
            self.csv_path = self.input_dir / "CheXpert-v1.0-small" / "train.csv"
        
        if not self.csv_path.exists():
            logger.error(f"Could not find train.csv in {input_dir}")
            sys.exit(1)
        
    def process(self):
        logger.info(f"Loading raw data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)

        # This removes "CheXpert-v1.0-small/" or "CheXpert-v1.0/" from the start of the path
        df['Path'] = df['Path'].str.replace(r'^CheXpert-v1\.0(-small)?/','',regex=True)

        logger.info(" Fixed file paths in CSV to match flattened directory structure")

        df['patient_id'] = df['Path'].apply(lambda x: x.split('/')[1])

        unique_patients = df['patient_id'].unique()
        logger.info(f"Total unique patients: {len(unique_patients)}")

        test_patient_ids = unique_patients[:10]
        df_test = df[df['patient_id'].isin(test_patient_ids)]
        self._save_split(df_test, "test.csv")
        logger.info(f" Created 'test.csv' with {len(df_test)} images (10 patients)")

        remaining_patients = list(set(unique_patients) - set(test_patient_ids))

        train_patients, val_patients = train_test_split(
            unique_patients, 
            test_size=0.20, 
            random_state=42
        )

        df_train = df[df['patient_id'].isin(train_patients)]
        df_val = df[df['patient_id'].isin(val_patients)]
        
        self._save_split(df_train, "train.csv")
        self._save_split(df_val, "val.csv")

    def _save_split(self, df: pd.DataFrame, filename: str):
        """Saves the dataframe and fixes image paths to be absolute or relative correctly."""

        save_path = self.output_dir / filename

        df_clean = df.fillna(0)

        df_clean.to_csv(save_path, index=False)

def main():
    parser = argparse.ArgumentParser(description='Preprocess CheXpert Metadata')
    parser.add_argument('--input', '-i', default='./data/raw', help='Path to raw data folder')
    parser.add_argument('--output', '-o', default='./data/processed', help='Path to save processed CSVs')
    args = parser.parse_args()

    processor = CheXpertPreprocessor(input_dir=args.input, output_dir=args.output)
    processor.process()

if __name__ == "__main__":
    main()