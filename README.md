# Bacterial Drug Resistance Predictor

## Project Overview

The **Bacterial Drug Resistance Predictor** is a machine learning pipeline built with FastAPI and TensorFlow/Keras designed to predict antibiotic resistance from bacterial genomic sequences (FASTA files). This application provides a web-based interface for users to upload genomic data, train/retrain models, validate model performance, and view real-time prediction results and model statistics.

## Features

* **Antibiotic Resistance Prediction:** Upload FASTA files containing bacterial genomic sequences to get predictions on resistance to various antibiotic classes.
* **Model Training:** Train a new deep learning model from scratch using a provided dataset (ZIP archive containing FASTA files and a `metadata.json` for labels).
* **Model Retraining:** Update an existing model with new data to improve its performance or adapt to new resistance patterns.
* **Model Validation:** Evaluate the loaded model's performance on a separate validation dataset, providing metrics like accuracy, F1-score (macro), and Hamming loss.
* **Interactive Dashboard:** A comprehensive dashboard to monitor model status, training statistics, recent predictions, frequent resistances, and real-time training progress (via chart).
* **Model Summary:** View the detailed architecture summary of the loaded Keras model.
* **Data Handling:** Utilizes `BioPython` for sequence parsing and `scikit-learn` for multi-label binarization of antibiotic classes.

## Technologies Used

* **Backend:**
    * [FastAPI](https://fastapi.tiangolo.com/): High-performance web framework for building APIs.
    * [TensorFlow / Keras](https://www.tensorflow.org/): Deep learning framework for building and training the resistance prediction model.
    * [Uvicorn](https://www.uvicorn.org/): ASGI server for running the FastAPI application.
* **Data Processing & ML Utilities:**
    * [BioPython](https://biopython.org/): For parsing FASTA sequence files.
    * [scikit-learn](https://scikit-learn.org/): For data preprocessing, including `MultiLabelBinarizer` for handling multiple antibiotic labels per sequence.
    * [NumPy](https://numpy.org/): Numerical computing.
    * [Pandas](https://pandas.pydata.org/): Data manipulation and analysis.
    * [Joblib](https://joblib.readthedocs.io/): For efficient saving and loading of Python objects (e.g., `MultiLabelBinarizer`).
* **Frontend:**
    * [HTML5](https://developer.mozilla.org/en-US/docs/Web/HTML)
    * [CSS3](https://developer.mozilla.org/en-US/docs/Web/CSS)
    * [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
    * [Bootstrap 5](https://getbootstrap.com/): Responsive UI framework.
    * [Chart.js](https://www.chartjs.org/): For interactive data visualization on the dashboard.
    * [Font Awesome](https://fontawesome.com/): Icons.

## Setup Instructions

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd bacterial-resistance-predictor
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is not provided, you would manually install the dependencies listed in `main.py`, `model.py`, `data_loader.py` such as `fastapi`, `uvicorn`, `tensorflow`, `biopython`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `matplotlib`, `seaborn`, `python-multipart`)*

### Running the Application

1.  **Start the FastAPI application:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    (The `--reload` flag is useful for development as it automatically restarts the server on code changes.)

2.  **Access the application:**
    Open your web browser and navigate to:
    * **Prediction Page:** `http://localhost:8000/`
    * **Dashboard:** `http://localhost:8000/dashboard`

## Data Preparation

To train or validate the model, you need a dataset in a specific format: a ZIP archive containing FASTA files and a `metadata.json` file. The `metadata.json` should map sequence IDs to their corresponding antibiotic resistance labels.

A helper script, `getiing_training_data.py`, is provided to demonstrate how to prepare training data from the [CARD (Comprehensive Antibiotic Resistance Database)](https://card.mcmaster.ca/) dataset.

### Steps to prepare CARD data using `getiing_training_data.py`:

1.  **Run the data preparation script:**
    ```bash
    python getiing_training_data.py
    ```
    This script will:
    * Download the CARD broadstreet archive.
    * Extract necessary FASTA and JSON files.
    * Parse the CARD JSON to map ARO (Antibiotic Resistance Ontology) terms to the project's predefined antibiotic classes.
    * Organize FASTA sequences into folders named after the antibiotic classes they are resistant to.
    * Create a `metadata.json` file that lists each sequence and its associated resistance labels.
    * Finally, it will create a `card_training_data.zip` file in the root directory, which is ready to be uploaded to the `/train` or `/validate` endpoints.

    **Note:** The mapping in `getiing_training_data.py` (`CARD_DRUG_CLASS_TO_OUR_CLASSES_MAP`) is a simplified example. For high accuracy, you might need to refine this mapping to precisely align CARD's ontology with your specific antibiotic classes, or consider using a more sophisticated ontology parsing library.

## API Endpoints

* **`/` (GET):** Serves the main prediction HTML page.
* **`/dashboard` (GET):** Serves the dashboard HTML page with model statistics and recent activities.
* **`/predict` (POST):**
    * **Input:** `fastaFile` (FASTA file).
    * **Output:** JSON object with predicted resistance probabilities for each sequence.
    * **Description:** Upload a FASTA file to predict antibiotic resistance using the currently loaded model.
* **`/train` (POST):**
    * **Input:** `file` (ZIP archive with training FASTA files and `metadata.json`), `epochs` (int), `test_size` (float).
    * **Output:** JSON message indicating training start and basic training info.
    * **Description:** Initializes and trains a new model from scratch.
* **`/retrain` (POST):**
    * **Input:** `file` (ZIP archive with new training FASTA files and `metadata.json`), `epochs` (int).
    * **Output:** JSON message indicating retraining completion and info.
    * **Description:** Retrains the existing model with additional data.
* **`/validate` (POST):**
    * **Input:** `file` (ZIP archive with validation FASTA files and `metadata.json`), `validation_name` (string).
    * **Output:** JSON message indicating validation completion and metrics.
    * **Description:** Evaluates the loaded model's performance on a given validation dataset.
* **`/model-summary` (GET):**
    * **Output:** JSON object containing the string summary of the Keras model.
    * **Description:** Provides the `model.summary()` output for the currently loaded model.
* **`/training-progress` (GET):**
    * **Output:** JSON object with real-time training history (epochs, loss, accuracy, etc.).
    * **Description:** Used by the dashboard to display the training progress chart.

## Usage Examples

### Via Web Interface

1.  **Predict:** Go to `http://localhost:8000/`, upload a FASTA file, and click "Predict". Results will be displayed and can be downloaded.
2.  **Train:** Go to `http://localhost:8000/dashboard`, use the "Initial Model Training" section to upload your prepared training ZIP file, set epochs and test size, then click "Start Training".
3.  **Retrain:** In the dashboard, use the "Model Retraining" section to upload new data and specify retraining epochs.
4.  **Validate:** In the dashboard, use the "Model Validation" section to upload a validation ZIP and provide a name for the run.
5.  **View Dashboard:** Navigate to `http://localhost:8000/dashboard` to see all the statistics, recent activities, and model details.

### Programmatic (Example using `curl`)

*(Replace `<your_zip_file.zip>` and `<your_fasta_file.fasta>` with actual paths)*

```bash
# Train a new model
curl -X POST "http://localhost:8000/train" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/training_data.zip;type=application/zip" \
  -F "epochs=50" \
  -F "test_size=0.2"

# Predict resistance from a FASTA file
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/sequence.fasta;type=text/plain"

# Get model summary
curl -X GET "http://localhost:8000/model-summary" \
  -H "accept: application/json"

# Validate model
curl -X POST "http://localhost:8000/validate" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/validation_data.zip;type=application/zip" \
  -F "validation_name=MyFirstValidation"