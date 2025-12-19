"""
CheXpert Dataset Downloader
Downloads and organizes the Stanford CheXpert dataset (40GB)
Supports both Kaggle API and manual download with resume capability
"""

import os
import sys
import argparse
import subprocess
import zipfile
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
import yaml
from common import setup_logger

logger = setup_logger("CheXpertDownloader")
class CheXpertManager():
    """
    Main downloader class for CheXpert dataset
    """
    DATASET_CONFIG = {
        'chexpert': {
            'kaggle_slug': 'ashery/chexpert',
            'zip_name': 'chexpert.zip',
            'expected_size_gb': 11.5,
            'dirs_to_check': ['train', 'valid'],
            'files_to_check': ['train.csv', 'valid.csv']
        }
    }
    def __init__(self, output_dir: str,use_kaggle: bool = True):
        """
        Initialize downloader
        
        Args:
            output_dir: Directory to save dataset
            use_kaggle: Use Kaggle API if available
        """

        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / 'raw'
        self.config = self.DATASET_CONFIG['chexpert']
        self.use_kaggle = use_kaggle

        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _check_and_fix_filenames(self) -> bool:
        """
        Universal Fix: Checks for 'archive.zip' (manual download default)
        and renames it to 'chexpert.zip' so the pipeline stays consistent.
        """

        standard_path = self.raw_dir / self.config['zip_name']
        manual_path = self.raw_dir / "archive.zip"

        if not standard_path.exists() and manual_path.exists():
            logger.info("Detected manual download ('archive.zip'). Renaming to standard format...")
            manual_path.rename(standard_path)
            return True
        return standard_path.exists()
    
    def download_kaggle(self) -> bool:
        """Downloads using the Kaggle API."""
        target_file = self.raw_dir / self.config['zip_name']
        
        if target_file.exists():
            logger.info(f"Archive already exists at {target_file}. Skipping download.")
            return True
        
        if not self.use_kaggle:
            return False
        
        logger.info(f"Starting Kaggle API download: {self.config['kaggle_slug']}")

        try:
            cmd = [
                'kaggle', 'datasets', 'download',
                '-d', self.config['kaggle_slug'],
                '-p', str(self.raw_dir)
            ]
            subprocess.run(cmd, check=True)

            self._check_and_fix_filenames()

            logger.info("Kaggle download completed.")
            return True
        
        except subprocess.CalledProcessError:
            logger.error("Kaggle download failed. Check your API credentials (kaggle.json).")
        except FileNotFoundError:
            logger.error("Kaggle CLI not found. Please install: pip install kaggle")

        return False
    
    def extract_and_organize(self) -> bool:
        """Extracts zip and flattens directory structure."""

        if self._validate_structure():
            logger.info("Dataset already extracted and validated. Skipping.")
            return True

        zip_path = self.raw_dir / self.config['zip_name']

        if not zip_path.exists():
            logger.error(f"Archive not found: {zip_path}")
            return False
        
        logger.info(f"Extracting {zip_path.name}...")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                file_list = zf.namelist()
                for member in tqdm(file_list, desc="Extracting", unit="files"):
                    zf.extract(member,self.raw_dir)

            extracted_items = [x for x in self.raw_dir.iterdir() if x.is_dir() and "chexpert" in x.name.lower()]

            if extracted_items:
                nested_dir = extracted_items[0]
                logger.info(f"Detected nested structure in {nested_dir.name}. Moving files up...")

                for item in nested_dir.iterdir():
                    shutil.move(str(item), str(self.raw_dir))
                
                nested_dir.rmdir()

            return self._validate_structure()
        
        except zipfile.BadZipFile:
            logger.error("The zip file is corrupted.")
            return False
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False
        
    def _validate_structure(self) -> bool:
        """Checks if expected files exist after extraction."""
        missing = []

        for d in self.config['dirs_to_check']:
            if not (self.raw_dir / d).exists():
                missing.append(d)

        for f in self.config['files_to_check']:
            if not (self.raw_dir / f).exists():
                missing.append(f)

        if missing:
            return False
        
        logger.info("Dataset structure validated successfully.")

        return True
    
    def run(self):
        """Main execution flow."""
        self._check_and_fix_filenames()

        if not (self.raw_dir / self.config['zip_name']).exists():
            success = self.download_kaggle()
            if not success:
                logger.error("Download failed or skipped")
                logger.info(f"PLEASE MANUALLY DOWNLOAD: https://www.kaggle.com/{self.config['kaggle_slug']}")
                logger.info(f"Place the file in: {self.raw_dir}")
                sys.exit(1)

            if self.extract_and_organize():
                logger.info(f"\n SUCCESS! Dataset is ready at: {self.raw_dir}")
            else:
                logger.error("\n FAILED. See logs for details.")

def main():
    parser = argparse.ArgumentParser(description="CheXpert Manager")
    parser.add_argument('--output', '-o', default='./data', help='Base data directory')
    parser.add_argument('--no-kaggle', action='store_true', help='Skip Kaggle API attempt')
    args = parser.parse_args()

    manager = CheXpertManager(output_dir=args.output, use_kaggle=not args.no_kaggle)
    manager.run()

if __name__ == "__main__":
    main()