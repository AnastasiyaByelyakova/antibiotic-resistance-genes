import mysql.connector
import os
import json
from dotenv import load_dotenv
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime # Import datetime for proper type hinting/usage
from passlib.hash import bcrypt # Import bcrypt for password hashing

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration from environment variables
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE")
}

def get_db_connection():
    """Establishes and returns a new database connection."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        logger.info("Successfully connected to the database.")
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Error connecting to database: {err}")
        # Depending on your error handling strategy, you might want to re-raise or return None
        raise ConnectionError(f"Database connection failed: {err}")

def close_db_connection(conn):
    """Closes the given database connection."""
    if conn and conn.is_connected():
        conn.close()
        logger.info("Database connection closed.")

# --- Authentication and User Management Functions ---

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Fetches a user from the database by username."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        sql = "SELECT id, username, hashed_password FROM users WHERE username = %s"
        cursor.execute(sql, (username,))
        user = cursor.fetchone()
        return user
    except mysql.connector.Error as err:
        logger.error(f"Error fetching user '{username}': {err}")
        return None
    finally:
        close_db_connection(conn)

def create_user(username: str, password: str) -> Optional[int]:
    """
    Creates a new user with a hashed password.
    Returns the new user's ID if successful, None otherwise.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Hash the password using bcrypt
        hashed_password = bcrypt.hash(password, rounds=12) # Use bcrypt for strong hashing

        sql = "INSERT INTO users (username, hashed_password) VALUES (%s, %s)"
        cursor.execute(sql, (username, hashed_password))
        conn.commit()
        logger.info(f"User '{username}' created successfully. ID: {cursor.lastrowid}")
        return cursor.lastrowid
    except mysql.connector.Error as err:
        if err.errno == 1062: # MySQL error code for duplicate entry (e.g., duplicate username)
            logger.warning(f"Attempted to create user '{username}', but username already exists.")
            raise ValueError("Username already exists.")
        logger.error(f"Error creating user '{username}': {err}")
        if conn:
            conn.rollback()
        raise
    finally:
        close_db_connection(conn)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password using bcrypt."""
    try:
        return bcrypt.verify(plain_password, hashed_password)
    except ValueError as e:
        logger.error(f"Error verifying password: {e}. Hashed password might be invalid.")
        return False


# --- Functions for Inserting Data ---

def insert_training_run(
    filename: str,
    epochs_trained: int,
    train_samples: int,
    test_samples: int,
    final_val_loss: Optional[float],
    final_val_accuracy: Optional[float],
    test_loss: Optional[float],
    test_accuracy: Optional[float],
    hamming_loss: Optional[float],
    jaccard_score: Optional[float],
    model_version: str,
    keras_version: str,
    tensorflow_version: str,
    classification_report: Dict[str, Any]
):
    """Inserts a new training run record into the database."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = """
        INSERT INTO training_runs (
            filename, epochs_trained, train_samples, test_samples,
            final_val_loss, final_val_accuracy, test_loss, test_accuracy,
            hamming_loss, jaccard_score, model_version, keras_version,
            tensorflow_version, classification_report
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        val = (
            filename, epochs_trained, train_samples, test_samples,
            final_val_loss, final_val_accuracy, test_loss, test_accuracy,
            hamming_loss, jaccard_score, model_version, keras_version,
            tensorflow_version, json.dumps(classification_report) # Store JSON as string
        )
        cursor.execute(sql, val)
        conn.commit()
        logger.info(f"Training run for '{filename}' recorded successfully. ID: {cursor.lastrowid}")
        return cursor.lastrowid
    except mysql.connector.Error as err:
        logger.error(f"Error inserting training run: {err}")
        if conn:
            conn.rollback() # Rollback on error
        raise
    finally:
        close_db_connection(conn)

def insert_retraining_run(
    filename: str,
    new_samples: int,
    epochs_retrained: int,
    final_loss: Optional[float],
    model_version: str
):
    """Inserts a new retraining run record into the database."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = """
        INSERT INTO retraining_runs (
            filename, new_samples, epochs_retrained, final_loss, model_version
        ) VALUES (%s, %s, %s, %s, %s)
        """
        val = (filename, new_samples, epochs_retrained, final_loss, model_version)
        cursor.execute(sql, val)
        conn.commit()
        logger.info(f"Retraining run for '{filename}' recorded successfully. ID: {cursor.lastrowid}")
        return cursor.lastrowid
    except mysql.connector.Error as err:
        logger.error(f"Error inserting retraining run: {err}")
        if conn:
            conn.rollback()
        raise
    finally:
        close_db_connection(conn)

def insert_validation_run(
    validation_name: str,
    filename: str,
    num_samples: int,
    accuracy: Optional[float],
    f1_macro: Optional[float],
    hamming_loss: Optional[float],
    report: Dict[str, Any]
):
    """Inserts a new validation run record into the database."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = """
        INSERT INTO validation_runs (
            validation_name, filename, num_samples, accuracy, f1_macro, hamming_loss, report
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        val = (
            validation_name, filename, num_samples, accuracy,
            f1_macro, hamming_loss, json.dumps(report) # Store JSON as string
        )
        cursor.execute(sql, val)
        conn.commit()
        logger.info(f"Validation run '{validation_name}' recorded successfully. ID: {cursor.lastrowid}")
        return cursor.lastrowid
    except mysql.connector.Error as err:
        logger.error(f"Error inserting validation run: {err}")
        if conn:
            conn.rollback()
        raise
    finally:
        close_db_connection(conn)

def insert_prediction(
    input_filename: str,
    sequence_id: str,
    predicted_resistances: List[str],
    probabilities: Dict[str, float]
):
    """Inserts a new prediction record into the database."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = """
        INSERT INTO predictions (
            input_filename, sequence_id, predicted_resistances, probabilities
        ) VALUES (%s, %s, %s, %s)
        """
        val = (
            input_filename, sequence_id,
            json.dumps(predicted_resistances), # Store list as JSON string
            json.dumps(probabilities) # Store dict as JSON string
        )
        cursor.execute(sql, val)
        conn.commit()
        logger.info(f"Prediction for sequence '{sequence_id}' from '{input_filename}' recorded successfully.")
        return cursor.lastrowid
    except mysql.connector.Error as err:
        logger.error(f"Error inserting prediction: {err}")
        if conn:
            conn.rollback()
        raise
    finally:
        close_db_connection(conn)

# --- Functions for Fetching Data ---

def get_recent_predictions(limit: int = 20) -> List[Dict[str, Any]]:
    """Fetches the most recent predictions from the database."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True) # Return results as dictionaries
        sql = """
        SELECT id, timestamp, input_filename AS filename, sequence_id,
               predicted_resistances, probabilities
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT %s
        """
        cursor.execute(sql, (limit,))
        records = cursor.fetchall()
        
        # Parse JSON columns back to Python objects
        for record in records:
            if record.get('predicted_resistances'):
                record['predicted_resistances'] = json.loads(record['predicted_resistances'])
            else:
                record['predicted_resistances'] = [] # Ensure it's a list even if empty/null
            if record.get('probabilities'):
                record['probabilities'] = json.loads(record['probabilities'])
            else:
                record['probabilities'] = {} # Ensure it's a dict even if empty/null
            record['total_resistances'] = len(record['predicted_resistances']) # Add derived field
        return records
    except mysql.connector.Error as err:
        logger.error(f"Error fetching recent predictions: {err}")
        return []
    finally:
        close_db_connection(conn)

def get_recent_validations(limit: int = 10) -> List[Dict[str, Any]]:
    """Fetches the most recent validation runs from the database."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        sql = """
        SELECT id, timestamp, validation_name AS name, filename, num_samples,
               accuracy, f1_macro, hamming_loss, report
        FROM validation_runs
        ORDER BY timestamp DESC
        LIMIT %s
        """
        cursor.execute(sql, (limit,))
        records = cursor.fetchall()
        
        # Parse JSON report string back to dictionary
        for record in records:
            if record.get('report'):
                try:
                    report_content = record['report']
                    if isinstance(report_content, str): # Handle cases where it's still a string
                        record['report'] = json.loads(report_content)
                    else: # Assume it's already dict if not string (e.g., from direct insert)
                        record['report'] = report_content
                except (json.JSONDecodeError, TypeError):
                    record['report'] = "Invalid JSON report data."
            else:
                record['report'] = {} # Ensure it's a dict even if empty/null

            # Ensure metrics are floats, handle N/A cases for dashboard rendering
            # Use 'N/A' string for presentation if value is None
            record['metrics'] = {
                'accuracy': record['accuracy'] if record['accuracy'] is not None else 'N/A',
                'f1_macro': record['f1_macro'] if record['f1_macro'] is not None else 'N/A',
                'hamming_loss': record['hamming_loss'] if record['hamming_loss'] is not None else 'N/A',
            }
        return records
    except mysql.connector.Error as err:
        logger.error(f"Error fetching recent validations: {err}")
        return []
    finally:
        close_db_connection(conn)

def get_model_metadata() -> Dict[str, Any]:
    """Fetches the current model metadata from the database."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        sql = "SELECT * FROM model_metadata ORDER BY last_updated DESC LIMIT 1"
        cursor.execute(sql)
        record = cursor.fetchone()
        
        if record:
            # Convert boolean from tinyint(1) if necessary
            record['model_loaded_status'] = bool(record['model_loaded'])
            if record.get('frequent_resistances'):
                record['frequent_resistances'] = json.loads(record['frequent_resistances'])
            else:
                record['frequent_resistances'] = {} # Ensure it's a dict even if empty/null

            # Align key names for dashboard consumption
            record['last_trained'] = record['last_updated'] # Align with old 'last_trained' key
            record['total_predictions'] = record['total_predictions_made'] # Align with old 'total_predictions' key
            # 'unique_files_predicted_list' was an in-memory list, not directly in DB, so not fetched here
            return record
        return {} # Return empty dict if no metadata found
    except mysql.connector.Error as err:
        logger.error(f"Error fetching model metadata: {err}")
        return {}
    finally:
        close_db_connection(conn)

def update_model_metadata(
    model_loaded: Optional[bool] = None,
    current_model_path: Optional[str] = None,
    label_encoder_path: Optional[str] = None,
    input_sequence_length: Optional[int] = None,
    antibiotic_labels_count: Optional[int] = None,
    total_params: Optional[int] = None,
    layers_count: Optional[int] = None,
    keras_version: Optional[str] = None,
    tensorflow_version: Optional[str] = None,
    total_predictions_made: Optional[int] = None,
    unique_files_predicted: Optional[int] = None,
    last_prediction_time: Optional[str] = None, # Store as ISO format string
    frequent_resistances: Optional[Dict[str, Any]] = None,
    train_samples: Optional[int] = None, # New field
    test_samples: Optional[int] = None,  # New field
    test_accuracy: Optional[float] = None, # To update dashboard test accuracy from latest training
    last_trained: Optional[str] = None # To explicitly set last_trained timestamp
):
    """
    Updates the model_metadata table. Inserts if no record exists, otherwise updates.
    This function will intelligently update only provided fields.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if a metadata record already exists
        cursor.execute("SELECT id FROM model_metadata LIMIT 1")
        existing_id_row = cursor.fetchone() # Fetchone returns (id,) or None

        updates = []
        values = []

        # Map function arguments to column names and prepare for update
        update_map = {
            "model_loaded": model_loaded,
            "current_model_path": current_model_path,
            "label_encoder_path": label_encoder_path,
            "input_sequence_length": input_sequence_length,
            "antibiotic_labels_count": antibiotic_labels_count,
            "total_params": total_params,
            "layers_count": layers_count,
            "keras_version": keras_version,
            "tensorflow_version": tensorflow_version,
            "total_predictions_made": total_predictions_made,
            "unique_files_predicted": unique_files_predicted,
            "last_prediction_time": last_prediction_time,
            "train_samples": train_samples, # New
            "test_samples": test_samples,   # New
            "test_accuracy": test_accuracy, # New
        }
        
        # 'frequent_resistances' and 'last_trained' (mapped to last_updated) need special handling
        if frequent_resistances is not None:
            update_map['frequent_resistances'] = json.dumps(frequent_resistances)
        if last_trained is not None: # This maps to the 'last_updated' column, which also has ON UPDATE CURRENT_TIMESTAMP
            update_map['last_updated'] = last_trained

        for col, val in update_map.items():
            if val is not None: # Only update if value is provided
                if col == 'frequent_resistances': # Already JSON dumped above
                    updates.append(f"{col} = %s")
                    values.append(val)
                elif col == 'last_updated': # Handle the explicit timestamp for last_trained
                     updates.append(f"{col} = %s")
                     values.append(val)
                else:
                    updates.append(f"{col} = %s")
                    values.append(val)

        if existing_id_row:
            existing_id = existing_id_row[0]
            if updates:
                sql = f"UPDATE model_metadata SET {', '.join(updates)} WHERE id = %s"
                values.append(existing_id)
                cursor.execute(sql, tuple(values))
                conn.commit()
                logger.info(f"Model metadata updated successfully for ID: {existing_id}.")
            else:
                logger.info("No new metadata provided for update.")
        else:
            # If no existing record, construct an INSERT statement
            # Need to ensure all columns (even if None) are in the insert
            # Fetch current default values for non-provided fields to ensure a valid INSERT
            default_metadata = {
                "model_loaded": False,
                "current_model_path": None,
                "label_encoder_path": None,
                "input_sequence_length": None,
                "antibiotic_labels_count": None,
                "total_params": None,
                "layers_count": None,
                "keras_version": None,
                "tensorflow_version": None,
                "total_predictions_made": 0,
                "unique_files_predicted": 0,
                "last_prediction_time": None,
                "frequent_resistances": json.dumps({}),
                "train_samples": None,
                "test_samples": None,
                "test_accuracy": None,
            }
            # Overlay provided values
            for col, val in update_map.items():
                if col == 'last_updated': # Skip this for initial insert as it's default CURRENT_TIMESTAMP
                    continue
                if val is not None:
                    if col == 'frequent_resistances':
                        default_metadata[col] = json.dumps(val) # Already dumped if passed
                    else:
                        default_metadata[col] = val

            cols = ', '.join(default_metadata.keys())
            placeholders = ', '.join(['%s'] * len(default_metadata))
            sql = f"INSERT INTO model_metadata ({cols}) VALUES ({placeholders})"
            insert_values = tuple(default_metadata.values())
            
            cursor.execute(sql, insert_values)
            conn.commit()
            logger.info(f"Initial model metadata inserted successfully. ID: {cursor.lastrowid}")
        
    except mysql.connector.Error as err:
        logger.error(f"Error updating model metadata: {err}")
        if conn:
            conn.rollback()
        raise
    finally:
        close_db_connection(conn)

def increment_total_predictions_and_update_last_time(filename: str):
    """
    Increments total predictions and updates last prediction time.
    Also updates unique files predicted.
    Frequent resistances must be handled by the caller of this function
    or update_model_metadata directly with an aggregated value.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get current metadata
        cursor.execute("SELECT id, total_predictions_made FROM model_metadata LIMIT 1")
        record = cursor.fetchone()

        if record is None:
            logger.warning("No model_metadata record found. Inserting initial record before updating prediction counts.")
            update_model_metadata(
                model_loaded=False,
                total_predictions_made=0,
                unique_files_predicted=0,
                frequent_resistances={}
            )
            cursor.execute("SELECT id, total_predictions_made FROM model_metadata LIMIT 1")
            record = cursor.fetchone()
            if record is None:
                raise RuntimeError("Failed to get or create model_metadata record.")

        db_id = record[0]
        total_predictions_made = record[1] + 1 if record[1] is not None else 1
        
        # Fetch all prediction filenames to accurately count unique files
        cursor.execute("SELECT DISTINCT input_filename FROM predictions")
        existing_prediction_filenames = {row[0] for row in cursor.fetchall()}
        unique_files_predicted = len(existing_prediction_filenames)

        sql = """
        UPDATE model_metadata
        SET total_predictions_made = %s,
            last_prediction_time = %s,
            unique_files_predicted = %s
        WHERE id = %s
        """
        val = (
            total_predictions_made,
            datetime.now().isoformat(), # Use current timestamp for last_prediction_time
            unique_files_predicted,
            db_id
        )
        cursor.execute(sql, val)
        conn.commit()
        logger.info("Model metadata (prediction counts) updated.")
    except mysql.connector.Error as err:
        logger.error(f"Error incrementing predictions/updating metadata: {err}")
        if conn:
            conn.rollback()
        raise
    finally:
        close_db_connection(conn)
