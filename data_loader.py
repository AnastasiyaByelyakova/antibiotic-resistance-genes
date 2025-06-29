import os
import requests
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import json
import zipfile
import logging
from typing import List, Dict, Tuple, Optional, Any
import io
import joblib # For saving/loading MultiLabelBinarizer
import tempfile # For creating temporary directories
import traceback # For detailed error logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.resistance_genes_db = {}
        self.antibiotic_classes = [
            'ampicillin', 'amoxicillin', 'ceftriaxone', 'ciprofloxacin',
            'tetracycline', 'chloramphenicol', 'gentamicin', 'streptomycin',
            'kanamycin', 'erythromycin', 'vancomycin', 'methicillin',
            'penicillin', 'trimethoprim', 'sulfamethoxazole'
        ]
        os.makedirs(data_dir, exist_ok=True)
        self.mlb = None # MultiLabelBinarizer instance

        # Try to load existing MultiLabelBinarizer
        try:
            self.mlb = joblib.load(os.path.join(self.data_dir, 'label_encoder.pkl'))
            logger.info("Existing MultiLabelBinarizer loaded successfully.")
        except FileNotFoundError:
            logger.warning("MultiLabelBinarizer not found. It will be fitted upon first training.")

    def sequence_to_features(self, sequence: str, target_length: int = 256) -> np.array:
        """
        Converts a DNA sequence into a fixed-length numerical feature vector.
        Pads/truncates sequences and uses one-hot encoding for nucleotides.
        """
        # Define nucleotide mapping (A:0, C:1, G:2, T:3) for simpler encoding
        nucleotide_map = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'T': 1.0, 'N': 0.0} # Using float values

        # Pad or truncate sequence to target_length
        if len(sequence) < target_length:
            padded_sequence = sequence + 'N' * (target_length - len(sequence))
        else:
            padded_sequence = sequence[:target_length]

        # Convert to numerical features
        features = np.array([nucleotide_map.get(base, 0.0) for base in padded_sequence])
        return features

    def load_and_preprocess_data(self, fasta_path: str = "data/sequences.fasta",
                                 metadata_path: str = "data/metadata.json"):
        """
        Loads and preprocesses data from local FASTA and JSON files.
        This method is primarily for local testing/initial data preparation,
        not directly used by the FastAPI upload endpoints.
        """
        if not os.path.exists(fasta_path) or not os.path.exists(metadata_path):
            logger.error(f"Required files not found: {fasta_path}, {metadata_path}")
            raise FileNotFoundError("FASTA or metadata file not found. Please ensure data is available.")

        sequences = []
        labels = []
        metadata = {}

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            # Assuming metadata contains a dictionary mapping sequence_id to resistances list
            sequence_resistances = {item['sequence_id']: item['resistances'] for item in metadata.get('samples', [])}

        # Load FASTA sequences
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_id = record.id
            if seq_id in sequence_resistances:
                sequences.append(str(record.seq))
                labels.append(sequence_resistances[seq_id])
            else:
                logger.warning(f"Sequence ID {seq_id} not found in metadata, skipping.")

        if not sequences:
            raise ValueError("No sequences loaded. Check FASTA and metadata files.")

        X = np.array([self.sequence_to_features(seq) for seq in sequences])

        # Fit and transform labels with MultiLabelBinarizer if not already fitted
        if self.mlb is None:
            self.mlb = MultiLabelBinarizer(classes=self.antibiotic_classes) # Ensure MLB uses a consistent order of classes
            y = self.mlb.fit_transform(labels)
            joblib.dump(self.mlb, os.path.join(self.data_dir, 'label_encoder.pkl')) # Save the fitted MLB
            logger.info("MultiLabelBinarizer fitted and saved.")
        else:
            y = self.mlb.transform(labels)
            logger.info("MultiLabelBinarizer transformed labels using existing encoder.")

        # Split data (default 80/20 train/test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info(f"Data loaded: {len(sequences)} samples. Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def _extract_zip_contents(self, zip_path: str, extract_dir: str):
        """Extracts the contents of a ZIP file to a specified directory."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"ZIP file extracted to {extract_dir}")
        except zipfile.BadZipFile:
            logger.error(f"Error: {zip_path} is not a valid ZIP file.")
            raise ValueError("The uploaded file is not a valid ZIP archive.")
        except Exception as e:
            logger.error(f"Error extracting ZIP file: {e}")
            raise Exception(f"Failed to extract ZIP file: {e}")

    def _load_metadata_from_zip_extract(self, extracted_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        Loads metadata.json if present and returns initial sequences_data structure.
        sequences_data: {sequence_id: {sequence_str: "...", classes: ["drug1", "drug2"]}}
        """
        sequences_data = {}
        metadata_json_path = os.path.join(extracted_dir, 'metadata.json')
        if os.path.exists(metadata_json_path):
            with open(metadata_json_path, 'r') as f:
                zip_metadata = json.load(f)
                if 'samples' in zip_metadata:
                    for item in zip_metadata['samples']:
                        seq_id = item['sequence_id']
                        sequences_data[seq_id] = {'sequence_str': None, 'classes': item['resistances']}
                    logger.info("metadata.json loaded from ZIP for explicit labeling.")
                return sequences_data, zip_metadata
        else:
            logger.info("No metadata.json found in ZIP. Will attempt labeling from folder structure.")
            return sequences_data, {}

    def _read_fasta_files_from_extracted_dir(self, extracted_dir: str, sequences_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Reads FASTA files from the extracted directory, updating sequences_data.
        Prioritizes metadata labels if available, otherwise uses folder names for classes.
        """
        fasta_files_found = False
        
        # Determine if the root of the extracted_dir contains FASTA files directly
        # or if it implies a class-based folder structure.
        root_contains_fasta = any(f.endswith(('.fasta', '.fa', '.fna')) for f in os.listdir(extracted_dir) if os.path.isfile(os.path.join(extracted_dir, f)))
        root_contains_dirs = any(os.path.isdir(os.path.join(extracted_dir, d)) for d in os.listdir(extracted_dir))

        # Scenario 1: Flat FASTA files in the root (legacy support or prediction-only)
        if root_contains_fasta and not root_contains_dirs:
            for file_name in os.listdir(extracted_dir):
                if file_name.endswith(('.fasta', '.fa', '.fna')):
                    fasta_path_in_zip = os.path.join(extracted_dir, file_name)
                    for record in SeqIO.parse(fasta_path_in_zip, "fasta"):
                        seq_id = record.id
                        if seq_id in sequences_data and sequences_data[seq_id]['classes'] is not None:
                            sequences_data[seq_id]['sequence_str'] = str(record.seq)
                        else:
                            # For now, if no explicit label or folder, skip for training/validation
                            logger.warning(f"Sequence ID {seq_id} from {file_name} has no explicit label from metadata or folder, skipping for training/validation.")
            fasta_files_found = True
        # Scenario 2: Class-based folder structure or mixed
        else:
            for root, dirs, files in os.walk(extracted_dir):
                # Only process subdirectories that are immediate children of extracted_dir
                # and interpret them as antibiotic classes
                if os.path.dirname(root) == extracted_dir and os.path.basename(root) in self.antibiotic_classes:
                    class_name = os.path.basename(root)
                    for file_name in os.listdir(root):
                        if file_name.endswith(('.fasta', '.fa', '.fna')):
                            fasta_path_in_zip = os.path.join(root, file_name)
                            for record in SeqIO.parse(fasta_path_in_zip, "fasta"):
                                seq_id = record.id
                                if seq_id not in sequences_data:
                                    sequences_data[seq_id] = {'sequence_str': str(record.seq), 'classes': []}
                                elif sequences_data[seq_id]['sequence_str'] is None: # Fill sequence if not already
                                    sequences_data[seq_id]['sequence_str'] = str(record.seq)
                                
                                # Add the class from the folder name
                                if class_name not in sequences_data[seq_id]['classes']:
                                    sequences_data[seq_id]['classes'].append(class_name)
                                fasta_files_found = True
                # Also check for FASTA files directly in the root of extracted_dir if no class folders are found
                elif root == extracted_dir and not root_contains_dirs and not fasta_files_found:
                     for file_name in files:
                        if file_name.endswith(('.fasta', '.fa', '.fna')):
                            fasta_path_in_zip = os.path.join(root, file_name)
                            for record in SeqIO.parse(fasta_path_in_zip, "fasta"):
                                seq_id = record.id
                                # If metadata provides explicit labels, use them
                                if seq_id in sequences_data and sequences_data[seq_id]['classes'] is not None:
                                    sequences_data[seq_id]['sequence_str'] = str(record.seq)
                                else:
                                    logger.warning(f"Sequence ID {seq_id} from {file_name} has no explicit label from metadata or folder, skipping.")
                            fasta_files_found = True

        if not fasta_files_found:
            raise ValueError("No FASTA files found in the uploaded ZIP file, or they are not in a recognized folder structure/metadata format.")
        
        return sequences_data

    def _preprocess_sequences_and_labels(self, sequences_data: Dict[str, Dict[str, Any]], is_training_data: bool):
        """
        Converts sequences to features and binarizes labels using MultiLabelBinarizer.
        """
        # Filter out sequences without a valid sequence string or no associated classes
        processed_sequences = []
        processed_labels = []
        for seq_id, data in sequences_data.items():
            if data['sequence_str'] and data['classes']:
                processed_sequences.append(data['sequence_str'])
                processed_labels.append(data['classes'])
            else:
                logger.warning(f"Sequence {seq_id} skipped due to missing sequence data or antibiotic classes.")

        if not processed_sequences:
            raise ValueError("No valid sequences with corresponding labels found after processing ZIP contents.")

        X = np.array([self.sequence_to_features(seq) for seq in processed_sequences])

        # Handle MultiLabelBinarizer (fit for training, transform for others)
        if is_training_data:
            # If MLB not initialized or needs re-fitting (e.g., if antibiotic classes change)
            if self.mlb is None or sorted(self.mlb.classes_) != sorted(self.antibiotic_classes):
                logger.info("Fitting new MultiLabelBinarizer for training data.")
                self.mlb = MultiLabelBinarizer(classes=self.antibiotic_classes) # Use consistent order
            y = self.mlb.fit_transform(processed_labels)
            self._save_label_encoder() # Save the fitted MLB
            logger.info("MultiLabelBinarizer fitted and saved.")
        else:
            if self.mlb is None:
                raise ValueError("MultiLabelBinarizer not fitted. Train a model first with is_training_data=True.")
            y = self.mlb.transform(processed_labels)
            logger.info("MultiLabelBinarizer transformed labels using existing encoder.")

        return X, y

    def _split_data(self, X: np.ndarray, y: np.ndarray, test_size: Optional[float]):
        """Splits the data into training and test sets, with stratification if possible."""
        if test_size is not None and 0 < test_size < 1:
            # Ensure enough samples for stratification
            if y.shape[0] > 1 and np.min(np.sum(y, axis=0)) > 1: # Check if each class has at least 2 samples
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            else:
                logger.warning("Not enough samples per class for stratification. Splitting without stratification.")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            logger.info(f"Data split: Train {len(X_train)}, Test {len(X_test)} samples.")
            return X_train, X_test, y_train, y_test
        else:
            logger.warning("Test size not provided or invalid for training data. Returning full dataset as training.")
            return X, None, y, None # Return X, y directly if no split

    def _save_label_encoder(self):
        """Saves the current MultiLabelBinarizer instance."""
        if self.mlb:
            encoder_path = os.path.join(self.data_dir, 'label_encoder.pkl')
            joblib.dump(self.mlb, encoder_path)
            logger.info(f"MultiLabelBinarizer saved to {encoder_path}")

    def load_data_from_uploaded_zip(
        self,
        zip_path: str,
        is_training_data: bool = False,
        test_size: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """
        Loads and preprocesses data from an uploaded ZIP file.
        The ZIP file is expected to contain a folder structure where
        each folder's name is an antibiotic class, containing FASTA files.
        A metadata.json file is optional; if present and contains 'samples',
        it will be prioritized for mapping sequence IDs to resistances.
        Otherwise, labeling is done by folder names.

        Args:
            zip_path (str): Path to the uploaded ZIP file.
            is_training_data (bool): If True, fits the MultiLabelBinarizer.
                                     If False, uses the existing MLB to transform.
            test_size (Optional[float]): Fraction of data for testing. Only used if is_training_data is True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
            X_train, X_test, y_train, y_test (if is_training_data=True and test_size is not None)
            or X, y (if for retraining/validation), and the metadata from the ZIP.
            Returns (None, None, None, None, {}) if data loading fails.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                self._extract_zip_contents(zip_path, temp_dir)

                sequences_data, zip_metadata = self._load_metadata_from_zip_extract(temp_dir)

                # This step will update sequences_data with actual sequence strings from FASTA files
                # and potentially add classes from folder names if metadata.json wasn't used for all.
                sequences_data = self._read_fasta_files_from_extracted_dir(temp_dir, sequences_data)

                X, y = self._preprocess_sequences_and_labels(sequences_data, is_training_data)

                if is_training_data:
                    X_train, X_test, y_train, y_test = self._split_data(X, y, test_size)
                    return X_train, X_test, y_train, y_test, zip_metadata
                else:
                    return X, None, y, None, zip_metadata

            except ValueError as e:
                logger.error(f"Data content error in ZIP: {e}")
                raise ValueError(f"Invalid data in ZIP file: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing uploaded ZIP: {traceback.format_exc()}")
                raise Exception(f"Failed to process uploaded ZIP file: {e}")

        # This part should ideally not be reached if exceptions are raised correctly
        return None, None, None, None, {}
