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
from typing import List, Dict, Tuple, Optional
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
        # This is a basic example; more complex feature extraction might be needed for real applications
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
        sequences_data = {} # {sequence_id: {sequence_str: "...", classes: ["drug1", "drug2"]}}
        zip_metadata = {}

        # Create a temporary directory to extract ZIP contents
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                logger.info(f"ZIP file extracted to {temp_dir}")

                # First, try to load metadata.json for explicit mapping (higher priority)
                metadata_json_path = os.path.join(temp_dir, 'metadata.json')
                if os.path.exists(metadata_json_path):
                    with open(metadata_json_path, 'r') as f:
                        zip_metadata = json.load(f)
                        if 'samples' in zip_metadata:
                            for item in zip_metadata['samples']:
                                seq_id = item['sequence_id']
                                # Placeholder for sequence string, will be filled from FASTA
                                sequences_data[seq_id] = {'sequence_str': None, 'classes': item['resistances']}
                            logger.info("metadata.json loaded from ZIP for explicit labeling.")
                else:
                    logger.info("No metadata.json found in ZIP. Will attempt labeling from folder structure.")

                # Process FASTA files and/or folder structure
                fasta_files_found = False
                
                # Iterate through top-level directories first for class-based labeling
                for root, dirs, files in os.walk(temp_dir):
                    # Skip the base temporary directory itself if it contains only other dirs/files
                    if root == temp_dir:
                        # Check for direct FASTA files in the root if no subfolders indicate classes
                        if not dirs: # If no subdirectories, assume flat FASTA files (legacy support)
                            for file_name in files:
                                if file_name.endswith(('.fasta', '.fa', '.fna')):
                                    fasta_path_in_zip = os.path.join(root, file_name)
                                    for record in SeqIO.parse(fasta_path_in_zip, "fasta"):
                                        seq_id = record.id
                                        # If metadata provided explicit labels for this ID, use them
                                        if seq_id in sequences_data and sequences_data[seq_id]['classes'] is not None:
                                            sequences_data[seq_id]['sequence_str'] = str(record.seq)
                                        # Otherwise, if no folder structure/metadata, assume unlabelled (problematic for training)
                                        else:
                                            # For now, if no explicit label or folder, skip for training/validation
                                            logger.warning(f"Sequence ID {seq_id} from {file_name} has no explicit label from metadata or folder, skipping.")
                                    fasta_files_found = True
                            continue # Move to next iteration of os.walk (subdirectories)

                    # Only process subdirectories that are immediate children of temp_dir
                    # and interpret them as antibiotic classes
                    if os.path.dirname(root) == temp_dir and os.path.basename(root) in self.antibiotic_classes:
                        class_name = os.path.basename(root)
                        for file_name in os.listdir(root):
                            if file_name.endswith(('.fasta', '.fa', '.fna')):
                                fasta_path_in_zip = os.path.join(root, file_name)
                                for record in SeqIO.parse(fasta_path_in_zip, "fasta"):
                                    seq_id = record.id
                                    if seq_id not in sequences_data:
                                        sequences_data[seq_id] = {'sequence_str': str(record.seq), 'classes': []}
                                    elif sequences_data[seq_id]['sequence_str'] is None:
                                        sequences_data[seq_id]['sequence_str'] = str(record.seq)
                                    
                                    # Add the class from the folder name
                                    if class_name not in sequences_data[seq_id]['classes']:
                                        sequences_data[seq_id]['classes'].append(class_name)
                                    fasta_files_found = True
                    
                if not fasta_files_found:
                    raise ValueError("No FASTA files found in the uploaded ZIP file, or they are not in a recognized folder structure/metadata format.")

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
                    joblib.dump(self.mlb, os.path.join(self.data_dir, 'label_encoder.pkl')) # Save fitted MLB
                    logger.info("MultiLabelBinarizer fitted and saved.")
                    
                    if test_size is not None and 0 < test_size < 1:
                        # Ensure enough samples for stratification
                        if y.shape[0] > 1 and np.min(np.sum(y, axis=0)) > 1: # Check if each class has at least 2 samples
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                        else:
                            logger.warning("Not enough samples per class for stratification. Splitting without stratification.")
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                        logger.info(f"Data split: Train {len(X_train)}, Test {len(X_test)} samples.")
                        return X_train, X_test, y_train, y_test, zip_metadata
                    else:
                        logger.warning("Test size not provided or invalid for training data. Returning full dataset as training.")
                        return X, None, y, None, zip_metadata # Return X, y directly if no split
                else:
                    if self.mlb is None:
                        raise ValueError("MultiLabelBinarizer not fitted. Train a model first with is_training_data=True.")
                    y = self.mlb.transform(processed_labels)
                    logger.info("MultiLabelBinarizer transformed labels using existing encoder.")
                    return X, None, y, None, zip_metadata # For validation/retraining, return X, y directly

            except zipfile.BadZipFile:
                logger.error(f"Error: {zip_path} is not a valid ZIP file.")
                raise ValueError("The uploaded file is not a valid ZIP archive.")
            except FileNotFoundError as e:
                logger.error(f"File system error during ZIP processing: {e}")
                raise ValueError(f"Required file not found inside ZIP or during processing: {e}")
            except ValueError as e:
                logger.error(f"Data content error in ZIP: {e}")
                raise ValueError(f"Invalid data in ZIP file: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing uploaded ZIP: {traceback.format_exc()}")
                raise Exception(f"Failed to process uploaded ZIP file: {e}")

        return None, None, None, None, {} # Should not reach here if exceptions are raised properly

if __name__ == "__main__":
    # Test the data loader (example usage, adapt paths as needed)
    loader = DataLoader()

    # Example 1: Test with local data (if you have data/sequences.fasta and data/metadata.json)
    try:
        X_train, X_test, y_train, y_test = loader.load_and_preprocess_data(
            fasta_path="data/sequences.fasta", metadata_path="data/metadata.json"
        )
        print(f"\nDataset loaded successfully from local files!")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Number of antibiotic classes: {y_train.shape[1]}")
    except FileNotFoundError:
        print("\nSkipping local data load test: 'data/sequences.fasta' or 'data/metadata.json' not found.")
    except Exception as e:
        print(f"\nError during local data load test: {e}")

    # Example 2: Simulate loading from an uploaded ZIP (you'd need to create a dummy zip for this structure)
    # To test this, manually create a ZIP file named 'dummy_training_data_folder_structure.zip'
    # with the following structure:
    # dummy_training_data_folder_structure.zip
    # └── ampicillin/
    #     ├── seq1.fasta (content: >seq1\nATGC...)
    #     └── seq2.fasta (content: >seq2\nTGCA...)
    # └── ciprofloxacin/
    #     ├── seq3.fasta (content: >seq3\nGCAT...)
    #     └── seq1.fasta (content: >seq1\nATGC...) # seq1 is resistant to both ampicillin and ciprofloxacin

    # Example of creating a dummy ZIP for testing (run this once to generate the file)
    dummy_zip_path_folder_structure = "uploads/dummy_training_data_folder_structure.zip"
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    with zipfile.ZipFile(dummy_zip_path_folder_structure, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Create dummy data for ampicillin
        zf.writestr('ampicillin/seq1.fasta', '>seq1\nATGCGTACGT')
        zf.writestr('ampicillin/seq2.fasta', '>seq2\nTGCAACGTAC')
        # Create dummy data for ciprofloxacin (note: seq1 is also here)
        zf.writestr('ciprofloxacin/seq3.fasta', '>seq3\nGCATCGTAGC')
        zf.writestr('ciprofloxacin/seq1.fasta', '>seq1\nATGCGTACGT') # seq1 is resistant to both

    try:
        print(f"\nSimulating load from uploaded ZIP with folder structure: {dummy_zip_path_folder_structure}")
        # When is_training_data is True, it will attempt to split into train/test
        X_train_zip, X_test_zip, y_train_zip, y_test_zip, zip_meta = loader.load_data_from_uploaded_zip(
            dummy_zip_path_folder_structure, is_training_data=True, test_size=0.5
        )
        print(f"Loaded from ZIP - Training set: {X_train_zip.shape}, Test set: {X_test_zip.shape}")
        print(f"ZIP Metadata (if any from metadata.json): {zip_meta}")
        print(f"Labels for first training sample: {loader.mlb.inverse_transform(y_train_zip[:1])}")
        print(f"Labels for first test sample: {loader.mlb.inverse_transform(y_test_zip[:1])}")

    except Exception as e:
        print(f"Error simulating ZIP load with folder structure: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_zip_path_folder_structure):
            os.remove(dummy_zip_path_folder_structure)
            print(f"Cleaned up {dummy_zip_path_folder_structure}")

