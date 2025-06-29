import os
import requests
import json
import zipfile
import shutil
import tarfile # New import for tar.gz extraction
from Bio import SeqIO
from collections import defaultdict
import logging
from typing import *
from pprint import pprint
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Updated URLs to point to the .tar.gz archive directly
CARD_ARCHIVE_URL = "https://card.mcmaster.ca/download/0/broadstreet-v4.0.1.tar.bz2"
CARD_ARCHIVE_NAME = "broadstreet-v4.0.1.tar.bz2" # Name of the archive file

# Paths within the extracted archive
CARD_JSON_IN_ARCHIVE = "card.json"
CARD_FASTA_IN_ARCHIVE = "nucleotide_fasta_protein_homolog_model.fasta"

# Replace with the actual list of antibiotic classes from your data_loader.py
OUR_ANTIBIOTIC_CLASSES = [
    'ampicillin', 'amoxicillin', 'ceftriaxone', 'ciprofloxacin',
    'tetracycline', 'chloramphenicol', 'gentamicin', 'streptomycin',
    'kanamycin', 'erythromycin', 'vancomycin', 'methicillin',
    'penicillin', 'trimethoprim', 'sulfamethoxazole'
]

DOWNLOAD_DIR = "card_downloads"
EXTRACT_DIR = os.path.join(DOWNLOAD_DIR, "extracted_card") # New directory for extracted files
PREPARED_DATA_DIR = "prepared_training_data"
OUTPUT_ZIP_NAME = "card_training_data.zip"

def download_file(url: str, dest_path: str) -> str:
    """Downloads a file from a URL to a specified destination path."""
    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded {os.path.basename(dest_path)}")
        return dest_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        raise

def extract_tar_gz(tar_gz_path: str, extract_to_dir: str):
    """Extracts a .tar.gz archive to a specified directory."""
    os.makedirs(extract_to_dir, exist_ok=True)
    logger.info(f"Extracting {tar_gz_path} to {extract_to_dir}...")
    try:
        with tarfile.open(tar_gz_path, "r:gz") as tar:
            tar.extractall(path=extract_to_dir)
        logger.info("Extraction complete.")
    except tarfile.ReadError as e:
        logger.error(f"Error reading tar.gz file {tar_gz_path}: {e}. It might be corrupted or not a valid tar.gz.")
        raise
    except Exception as e:
        logger.error(f"Error during extraction of {tar_gz_path}: {e}")
        raise

def get_aro_drug_class_mapping(card_json_path: str) -> Dict[str, List[str]]:
    """
    Parses card.json to create a mapping from ARO IDs to the high-level drug classes.
    This mapping is simplified to match OUR_ANTIBIOTIC_CLASSES.
    """
    logger.info(f"Parsing CARD JSON from {card_json_path}...")
    with open(card_json_path, 'r') as f:
        card_data = json.load(f)

    aro_to_our_classes = defaultdict(set) # ARO_ID -> set of our classes

    # Build a reverse map for quick lookup of our specific classes
    # This is a manual mapping and might need careful refinement based on CARD's ontology
    # and how accurately you want to map.
    # Key: Broad CARD drug class name, Value: List of our specific antibiotic classes
    with open('antibiotics_classes.json', 'r') as jh:
        CARD_DRUG_CLASS_TO_OUR_CLASSES_MAP = json.loads(jh.read())
    logger.info("Building ARO to simplified antibiotic class mapping...")
    for aro_id, aro_details in card_data.items():
        # Check for direct drug class terms
        if type(aro_details)==str:
            continue
        for a, b in aro_details.get('ARO_category', {}).items():
            drug_class_name = b['category_aro_name']   
            try:
                drug_class_aro_id =  list(list(aro_details['model_sequences'].values())[0].values())[0][ 'dna_sequence']['accession']
            except:
                continue
            normalized_name = drug_class_name.lower()
            if normalized_name in CARD_DRUG_CLASS_TO_OUR_CLASSES_MAP:
                for our_class in CARD_DRUG_CLASS_TO_OUR_CLASSES_MAP[normalized_name]:
                    if our_class in OUR_ANTIBIOTIC_CLASSES:
                        aro_to_our_classes[drug_class_aro_id].add(our_class)

            # Also check if the drug_class_name itself is directly in our classes (less likely for CARD's broad terms)
            if normalized_name in OUR_ANTIBIOTIC_CLASSES:
                aro_to_our_classes[drug_class_aro_id].add(normalized_name)

        # Consider 'is_a' relationships for broader classification if necessary
        # This part requires traversing the ontology tree, which can be complex.
        # For simplicity, we primarily rely on direct drug_class_aro_terms.
        # Example of adding via 'is_a' for "resistance mechanism" parent AROs
        for parent_aro_id in aro_details.get('is_a', []):
            if parent_aro_id in card_data:
                parent_details = card_data[parent_aro_id]
                parent_name = parent_details.get('name', '').lower()
                
                if parent_name in CARD_DRUG_CLASS_TO_OUR_CLASSES_MAP:
                    for our_class in CARD_DRUG_CLASS_TO_OUR_CLASSES_MAP[parent_name]:
                        if our_class in OUR_ANTIBIOTIC_CLASSES:
                            aro_to_our_classes[aro_id].add(our_class)
    
    # Convert sets to lists
    final_mapping = {aro: list(classes) for aro, classes in aro_to_our_classes.items()}
    logger.info(f"Generated ARO mapping for {len(final_mapping)} AROs.")
    return final_mapping

def prepare_training_data(fasta_path: str, aro_mapping: Dict[str, List[str]], output_dir: str):
    """
    Reads FASTA file, organizes sequences into antibiotic class folders.
    """
    logger.info(f"Preparing training data in {output_dir}...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # Clear previous run
    os.makedirs(output_dir)

    for class_name in OUR_ANTIBIOTIC_CLASSES:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    fasta_records_processed = 0
    genes_mapped_to_classes = 0
    
    # Track sequence IDs that have been written to avoid duplicates if a gene maps to multiple classes
    # and is written to multiple folders
    written_sequence_ids = defaultdict(set) 
    with open(fasta_path, 'r') as f:
        for record in SeqIO.parse(f, "fasta"):
            fasta_records_processed += 1
            # Extract ARO ID from description, e.g., "ARO:3000001 gene_name [species]"
            aro_id_part = record.description.split('|')[1]
            aro_id_part = aro_id_part.replace('ARO:','')
            if aro_id_part:
                associated_classes = aro_mapping.get(aro_id_part, [])
                if associated_classes:
                    genes_mapped_to_classes += 1
                    for class_name in associated_classes:
                        if class_name in OUR_ANTIBIOTIC_CLASSES: # Ensure it's one of our target classes
                            class_folder = os.path.join(output_dir, class_name)
                            # Create a unique filename for the sequence
                            gene_filename = f"{record.id}.fasta"
                            output_fasta_path = os.path.join(class_folder, gene_filename)
                            
                            # Only write if this sequence hasn't been written to this specific folder yet
                            if record.id not in written_sequence_ids[class_name]:
                                try:
                                    with open(output_fasta_path, "a") as out_f: # Use "a" to append
                                        SeqIO.write(record, out_f, "fasta")
                                    written_sequence_ids[class_name].add(record.id)
                                except:
                                    pass
    
    logger.info(f"Processed {fasta_records_processed} FASTA records.")
    logger.info(f"Mapped {genes_mapped_to_classes} genes to antibiotic classes.")
    logger.info(f"Training data prepared in '{output_dir}'")

def create_zip_archive(source_dir: str, output_zip: str):
    """Creates a ZIP archive from the source directory."""
    logger.info(f"Creating ZIP archive: {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Arcname is the path inside the zip file (relative to source_dir)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    logger.info(f"ZIP archive created successfully: {output_zip}")

if __name__ == "__main__":
    # Clean up previous downloads and prepared data
    # if os.path.exists(DOWNLOAD_DIR):
    #     shutil.rmtree(DOWNLOAD_DIR)
    # if os.path.exists(PREPARED_DATA_DIR):
    #     shutil.rmtree(PREPARED_DATA_DIR)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)


    # # 1. Download the CARD .tar.gz archive
    # card_archive_path = download_file(CARD_ARCHIVE_URL, os.path.join(DOWNLOAD_DIR, CARD_ARCHIVE_NAME))

    # # 2. Extract the contents from the .tar.gz archive
    # extract_tar_gz(card_archive_path, EXTRACT_DIR)
    
    # Set paths to the extracted files
    card_json_path = os.path.join(EXTRACT_DIR, CARD_JSON_IN_ARCHIVE)
    card_fasta_path = os.path.join(EXTRACT_DIR, CARD_FASTA_IN_ARCHIVE)

    # Check if extracted files exist
    if not os.path.exists(card_json_path):
        logger.error(f"Extracted card.json not found at {card_json_path}. Extraction might have failed or path is wrong.")
        exit(1)
    if not os.path.exists(card_fasta_path):
        logger.error(f"Extracted FASTA file not found at {card_fasta_path}. Extraction might have failed or path is wrong.")
        exit(1)

    # 3. Get ARO to our antibiotic classes mapping
    aro_mapping = get_aro_drug_class_mapping(card_json_path)

    # 4. Prepare training data in the required folder structure
    prepare_training_data(card_fasta_path, aro_mapping, PREPARED_DATA_DIR)

    # 5. Create the ZIP archive
    create_zip_archive(PREPARED_DATA_DIR, OUTPUT_ZIP_NAME)

    logger.info("\n--- Data Preparation Complete ---")
    logger.info(f"Your training data is ready in '{OUTPUT_ZIP_NAME}'.")
    logger.info(f"You can now upload this '{OUTPUT_ZIP_NAME}' file to your FastAPI dashboard for model training/retraining.")
    logger.info("Remember that the mapping in the script ('CARD_DRUG_CLASS_TO_OUR_CLASSES_MAP') is a simplified example. "
                "For high accuracy, you might need to refine this mapping "
                "to precisely align CARD's ontology with your specific antibiotic classes, "
                "or consider using a more sophisticated ontology parsing library.")

    # Clean up downloaded and prepared data directories (optional, enable after successful run)
    # shutil.rmtree(DOWNLOAD_DIR)
    # shutil.rmtree(PREPARED_DATA_DIR)
    # logger.info("Cleaned up temporary directories.")