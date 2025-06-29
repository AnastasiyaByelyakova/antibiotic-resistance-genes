from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
import requests
import json
import os
import logging
from datetime import datetime, timedelta

# Assuming database.py and data_loader.py are accessible via sys.path or Docker bind mounts
# For a real Airflow deployment, these would need to be part of the Airflow environment
# or imported from a shared volume/package.
# For simplicity, we'll assume they are accessible directly or have paths configured.

# Import local modules (adjust sys.path if necessary in your Airflow setup)
try:
    from database import execute_query
    from data_loader import DataLoader
except ImportError:
    # This block is for Airflow environments where local modules might not be in sys.path
    # In a production Airflow, you'd typically package your code or mount volumes.
    import sys
    # Adjust this path based on where your app code lives relative to the DAGs folder
    # Example: if your DAGs are in /opt/airflow/dags and your app in /opt/airflow/app
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app_code_directory')))
    from database import execute_query
    from data_loader import DataLoader
    logging.info("Imported local modules successfully within Airflow context.")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Airflow Variables ---
# It's best practice to store sensitive info (like API base URL, thresholds, emails) in Airflow Variables
# You would set these in Airflow UI -> Admin -> Variables
# Example:
# Key: fastapi_app_url, Value: http://host.docker.internal:8000 (if Airflow is in Docker, host might be different)
# Key: accuracy_threshold, Value: 0.96
# Key: validation_zip_file_path, Value: /path/to/your/validation_data.zip
# Key: retraining_zip_file_path, Value: /path/to/your/retraining_data.zip
# Key: owner_email, Value: your_email@example.com

# For demonstration, using environment variables or hardcoding if Airflow Variables aren't set
FASTAPI_APP_URL = os.getenv('FASTAPI_APP_URL', 'http://host.docker.internal:8000') # Or http://localhost:8000
ACCURACY_THRESHOLD = float(os.getenv('ACCURACY_THRESHOLD', '0.96'))

# These paths would need to be accessible from within the Airflow worker/container
# They should point to pre-existing ZIP files that contain validation/retraining data.
VALIDATION_ZIP_FILE_PATH = os.getenv('VALIDATION_ZIP_FILE_PATH', '/opt/airflow/dags/data/validation_data.zip') # Example path
RETRAINING_ZIP_FILE_PATH = os.getenv('RETRAINING_ZIP_FILE_PATH', '/opt/airflow/dags/data/training_data.zip') # Example path

OWNER_EMAIL = os.getenv('OWNER_EMAIL', 'admin@example.com')

# --- DAG Definition ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': [OWNER_EMAIL],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='bacterial_resistance_model_automation',
    default_args=default_args,
    description='Automated model validation and retraining for bacterial drug resistance predictor.',
    schedule_interval=timedelta(weeks=1), # Run weekly
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'validation', 'retraining'],
) as dag:

    def _fetch_latest_data_version(is_training_data: bool):
        """Fetches the latest data version ID from the database."""
        logger.info(f"Fetching latest data version for {'training' if is_training_data else 'validation'}...")
        data_loader_instance = DataLoader() # Initialize DataLoader to use its DB methods
        latest_id = data_loader_instance.get_latest_data_version_id(is_training_data)
        if latest_id:
            logger.info(f"Found latest data version ID: {latest_id}")
            return latest_id
        else:
            raise ValueError(f"No latest data version found for is_training_data={is_training_data}. Please upload data first.")

    fetch_latest_validation_data_version = PythonOperator(
        task_id='fetch_latest_validation_data_version',
        python_callable=_fetch_latest_data_version,
        op_kwargs={'is_training_data': False},
    )

    fetch_latest_training_data_version = PythonOperator(
        task_id='fetch_latest_training_data_version',
        python_callable=_fetch_latest_data_version,
        op_kwargs={'is_training_data': True},
    )

    def _trigger_model_validation(ti, validation_zip_path):
        """Triggers the /validate API endpoint."""
        logger.info(f"Triggering model validation with {validation_zip_path}...")
        
        if not os.path.exists(validation_zip_path):
            raise FileNotFoundError(f"Validation ZIP file not found at: {validation_zip_path}")

        validation_name = f"Automated_Weekly_Validation_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        with open(validation_zip_path, 'rb') as f:
            files = {'file': (os.path.basename(validation_zip_path), f, 'application/zip')}
            data = {'validation_name': validation_name}
            
            try:
                response = requests.post(f"{FASTAPI_APP_URL}/validate", files=files, data=data, timeout=300) # 5 min timeout
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                result = response.json()
                logger.info(f"Validation API response: {result}")
                
                # Push validation accuracy to XCom for conditional retraining
                validation_accuracy = result.get('validation_results', {}).get('accuracy')
                if validation_accuracy is None:
                    raise ValueError("Validation accuracy not found in API response.")
                ti.xcom_push(key='validation_accuracy', value=validation_accuracy)
                logger.info(f"Validation accuracy pushed to XCom: {validation_accuracy}")
                return validation_accuracy

            except requests.exceptions.RequestException as e:
                logger.error(f"Error calling /validate API: {e}")
                raise e # Re-raise to mark task as failed
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from /validate API response: {response.text}")
                raise ValueError("Invalid JSON response from validation API.")

    trigger_model_validation = PythonOperator(
        task_id='trigger_model_validation',
        python_callable=_trigger_model_validation,
        op_kwargs={'validation_zip_path': VALIDATION_ZIP_FILE_PATH},
    )

    def _check_accuracy_and_branch(ti):
        """Checks validation accuracy and decides whether to retrain."""
        validation_accuracy = ti.xcom_pull(task_ids='trigger_model_validation', key='validation_accuracy')
        logger.info(f"Retrieved validation accuracy: {validation_accuracy}")
        
        if validation_accuracy is None:
            logger.error("Validation accuracy not found. Skipping retraining check.")
            return 'do_nothing' # Or raise error if this should always trigger
        
        if validation_accuracy < ACCURACY_THRESHOLD:
            logger.warning(f"Accuracy ({validation_accuracy:.4f}) is below threshold ({ACCURACY_THRESHOLD}). Retraining model.")
            return 'trigger_model_retraining'
        else:
            logger.info(f"Accuracy ({validation_accuracy:.4f}) is above or equal to threshold ({ACCURACY_THRESHOLD}). No retraining needed.")
            return 'do_nothing'

    check_accuracy_and_branch = BranchPythonOperator(
        task_id='check_accuracy_and_branch',
        python_callable=_check_accuracy_and_branch,
    )

    def _trigger_model_retraining(ti, retraining_zip_path):
        """Triggers the /retrain API endpoint."""
        logger.info(f"Triggering model retraining with {retraining_zip_path}...")

        if not os.path.exists(retraining_zip_path):
            raise FileNotFoundError(f"Retraining ZIP file not found at: {retraining_zip_path}")
            
        with open(retraining_zip_path, 'rb') as f:
            files = {'file': (os.path.basename(retraining_zip_path), f, 'application/zip')}
            data = {'epochs': 10} # Retrain for 10 epochs, adjust as needed

            try:
                response = requests.post(f"{FASTAPI_APP_URL}/retrain", files=files, data=data, timeout=600) # 10 min timeout
                response.raise_for_status()
                result = response.json()
                logger.info(f"Retraining API response: {result}")
                # You can push retraining info to XCom if needed for subsequent tasks
            except requests.exceptions.RequestException as e:
                logger.error(f"Error calling /retrain API: {e}")
                raise e

    trigger_model_retraining = PythonOperator(
        task_id='trigger_model_retraining',
        python_callable=_trigger_model_retraining,
        op_kwargs={'retraining_zip_path': RETRAINING_ZIP_FILE_PATH},
    )

    def _do_nothing():
        """Placeholder task when no retraining is needed."""
        logger.info("No retraining performed.")

    do_nothing = PythonOperator(
        task_id='do_nothing',
        python_callable=_do_nothing,
    )

    # --- Task Dependencies ---
    trigger_model_validation >> check_accuracy_and_branch
    check_accuracy_and_branch >> [trigger_model_retraining, do_nothing]

