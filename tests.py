import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os
import json
import numpy as np

# Import the main FastAPI app and other components
from main import app, data_loader, predictor, recent_predictions, recent_validations, current_training_history_file

# Create a TestClient for the FastAPI application
client = TestClient(app)

# --- Mocks Setup ---
# Use pytest fixtures for mocking to ensure clean state for each test

@pytest.fixture
def mock_data_loader():
    """Mocks the DataLoader instance."""
    with patch('main.DataLoader') as mock_loader_class:
        mock_loader = mock_loader_class.return_value
        # Configure mock_loader methods as needed for tests
        mock_loader.data_dir = "test_data_dir" # Mock data directory
        yield mock_loader

@pytest.fixture
def mock_predictor():
    """Mocks the ResistancePredictor instance."""
    with patch('main.ResistancePredictor') as mock_predictor_class:
        mock_pred = mock_predictor_class.return_value
        # Simulate a loaded model
        mock_pred.model = MagicMock()
        mock_pred.model.input_shape = (None, 100) # Example input shape
        mock_pred.model.output_shape = (None, 10)  # Example output shape (number of classes)
        mock_pred.model.count_params.return_value = 100000
        mock_pred.model.layers = [MagicMock()] * 5 # Example number of layers
        yield mock_pred

# --- Helper Functions for Tests ---
def create_dummy_fasta_file(filename="test.fasta", content=">seq1\nATGCGT\n>seq2\nTGCA"):
    """Creates a dummy FASTA file for testing."""
    filepath = os.path.join("uploads", filename)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath

def create_dummy_zip_file(filename="test_data.zip", content_dict=None):
    """Creates a dummy ZIP file with specified content."""
    import zipfile
    filepath = os.path.join("uploads", filename)
    if content_dict is None:
        content_dict = {
            "ampicillin/seqA.fasta": ">seqA\nATGC",
            "ciprofloxacin/seqB.fasta": ">seqB\nTGCA",
            "metadata.json": json.dumps({"description": "Test data"})
        }
    
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, data in content_dict.items():
            zf.writestr(name, data)
    return filepath

# --- Tests for FastAPI Endpoints ---

@pytest.fixture(autouse=True)
def run_around_tests(mock_data_loader, mock_predictor):
    """
    This fixture runs automatically before and after each test.
    It ensures that `data_loader` and `predictor` are mocked
    and that global states (like recent_predictions) are reset.
    """
    # Setup: Ensure mocks are in place before the app starts up
    # The `with patch(...)` in the fixture already handles this.
    
    # Manually call startup event to ensure `data_loader` and `predictor` are initialized
    # based on the mocked classes.
    with patch('main.data_loader', mock_data_loader), \
         patch('main.predictor', mock_predictor):
        # Reset global lists before each test
        recent_predictions.clear()
        recent_validations.clear()

        # Ensure the current_training_history_file is reset and its dummy content created/cleared
        global current_training_history_file
        test_history_path = os.path.join(mock_data_loader.data_dir, "current_training_history.json")
        current_training_history_file = test_history_path
        
        # Ensure the test_data_dir exists and is clean
        os.makedirs(mock_data_loader.data_dir, exist_ok=True)
        if os.path.exists(current_training_history_file):
            os.remove(current_training_history_file)
        with open(current_training_history_file, 'w') as f:
            json.dump({'epochs': [], 'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}, f)


        yield # Run the test

        # Teardown: Clean up any files created during the test
        if os.path.exists("uploads"):
            for f in os.listdir("uploads"):
                os.remove(os.path.join("uploads", f))
            os.rmdir("uploads")
        if os.path.exists(mock_data_loader.data_dir):
            for f in os.listdir(mock_data_loader.data_dir):
                os.remove(os.path.join(mock_data_loader.data_dir, f))
            os.rmdir(mock_data_loader.data_dir) # Clean up the mocked data_dir

def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Bacterial Drug Resistance Predictor" in response.text

def test_predict_success(mock_predictor, mock_data_loader):
    """Test successful prediction."""
    mock_predictor.model.predict.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
    mock_predictor.label_encoder = MagicMock()
    mock_predictor.label_encoder.classes_ = ['drugA', 'drugB']
    mock_data_loader.sequence_to_features.return_value = np.zeros(100) # Dummy feature vector

    # Ensure model is "loaded" for the test
    mock_predictor.load_model.return_value = None # This is called at startup

    fasta_content = ">seq1\nATGC\n>seq2\nTGCA"
    filepath = create_dummy_fasta_file(content=fasta_content)

    with open(filepath, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": (os.path.basename(filepath), f, "text/plain")}
        )

    assert response.status_code == 200
    assert response.json() == {
        "seq1": {"drugA": 0.9, "drugB": 0.1},
        "seq2": {"drugA": 0.2, "drugB": 0.8}
    }
    assert len(recent_predictions) == 2 # Two predictions should be added
    assert recent_predictions[0]['predicted_resistances'] == ['drugB'] # Check for seq2 based on 0.5 threshold
    assert recent_predictions[1]['predicted_resistances'] == ['drugA'] # Check for seq1 based on 0.5 threshold
    mock_predictor.predict_from_fasta.assert_called_once()

def test_predict_no_model_loaded(mock_predictor):
    """Test prediction when no model is loaded."""
    mock_predictor.model = None # Simulate no model loaded
    filepath = create_dummy_fasta_file()

    with open(filepath, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": (os.path.basename(filepath), f, "text/plain")}
        )
    assert response.status_code == 400
    assert "Model is not trained/loaded" in response.json()["detail"]

def test_train_success(mock_predictor, mock_data_loader):
    """Test successful model training."""
    zip_path = create_dummy_zip_file()

    # Mock the return values for data loading and training
    mock_data_loader.load_data_from_uploaded_zip.return_value = (
        np.zeros((10, 100)), np.zeros((2, 100)), np.zeros((10, 10)), np.zeros((2, 10)), {}
    )
    mock_predictor.train.return_value = {
        "timestamp": "2024-01-01T12:00:00",
        "epochs_trained": 10,
        "final_val_loss": 0.1,
        "final_val_accuracy": 0.9,
        "test_loss": 0.05,
        "test_accuracy": 0.95,
        "classification_report": {},
        "hamming_loss": 0.01,
        "jaccard_score": 0.9
    }

    with open(zip_path, "rb") as f:
        response = client.post(
            "/train",
            files={"file": (os.path.basename(zip_path), f, "application/zip")},
            data={"epochs": 1, "test_size": 0.2}
        )

    assert response.status_code == 200
    assert "Model training initiated successfully!" in response.json()["message"]
    mock_data_loader.load_data_from_uploaded_zip.assert_called_once()
    mock_predictor.train.assert_called_once()

def test_train_invalid_data(mock_predictor, mock_data_loader):
    """Test training with invalid data (e.g., empty data from loader)."""
    zip_path = create_dummy_zip_file()

    mock_data_loader.load_data_from_uploaded_zip.return_value = (
        None, None, None, None, {} # Simulate failure to load data
    )

    with open(zip_path, "rb") as f:
        response = client.post(
            "/train",
            files={"file": (os.path.basename(zip_path), f, "application/zip")},
            data={"epochs": 1, "test_size": 0.2}
        )
    assert response.status_code == 400
    assert "Failed to load and preprocess data from ZIP" in response.json()["detail"]

def test_retrain_success(mock_predictor, mock_data_loader):
    """Test successful model retraining."""
    zip_path = create_dummy_zip_file()

    # Ensure a model is "loaded" before retraining
    mock_predictor.model = MagicMock()
    mock_data_loader.load_data_from_uploaded_zip.return_value = (
        np.zeros((5, 100)), None, np.zeros((5, 10)), None, {}
    )
    mock_predictor.retrain.return_value = {
        "timestamp": "2024-01-01T13:00:00",
        "new_samples": 5,
        "epochs_retrained": 5,
        "final_loss": 0.02
    }

    with open(zip_path, "rb") as f:
        response = client.post(
            "/retrain",
            files={"file": (os.path.basename(zip_path), f, "application/zip")},
            data={"epochs": 5}
        )
    assert response.status_code == 200
    assert "Model retraining completed successfully!" in response.json()["message"]
    mock_predictor.retrain.assert_called_once()

def test_retrain_no_model_loaded(mock_predictor):
    """Test retraining when no model is loaded."""
    mock_predictor.model = None # Simulate no model loaded
    filepath = create_dummy_zip_file()

    with open(filepath, "rb") as f:
        response = client.post(
            "/retrain",
            files={"file": (os.path.basename(filepath), f, "application/zip")},
            data={"epochs": 5}
        )
    assert response.status_code == 400
    assert "No model loaded" in response.json()["detail"]

def test_validate_success(mock_predictor, mock_data_loader):
    """Test successful model validation."""
    zip_path = create_dummy_zip_file()

    mock_predictor.model = MagicMock() # Ensure a model is "loaded"
    mock_data_loader.load_data_from_uploaded_zip.return_value = (
        np.zeros((3, 100)), None, np.zeros((3, 10)), None, {}
    )
    mock_predictor.evaluate_model.return_value = {
        "accuracy": 0.98,
        "f1_macro": 0.97,
        "hamming_loss": 0.01,
        "jaccard_score": 0.95,
        "report": "{...classification report...}"
    }

    with open(zip_path, "rb") as f:
        response = client.post(
            "/validate",
            files={"file": (os.path.basename(zip_path), f, "application/zip")},
            data={"validation_name": "TestValidationRun"}
        )
    assert response.status_code == 200
    assert "Model validation completed successfully!" in response.json()["message"]
    assert len(recent_validations) == 1
    assert recent_validations[0]['name'] == "TestValidationRun"
    mock_predictor.evaluate_model.assert_called_once()

def test_validate_no_model_loaded(mock_predictor):
    """Test validation when no model is loaded."""
    mock_predictor.model = None # Simulate no model loaded
    filepath = create_dummy_zip_file()

    with open(filepath, "rb") as f:
        response = client.post(
            "/validate",
            files={"file": (os.path.basename(filepath), f, "application/zip")},
            data={"validation_name": "TestValidationRun"}
        )
    assert response.status_code == 400
    assert "No model loaded" in response.json()["detail"]

def test_get_model_summary_success(mock_predictor):
    """Test retrieving model summary when model is loaded."""
    mock_predictor.get_model_summary_string.return_value = "Mock Model Summary"
    response = client.get("/model-summary")
    assert response.status_code == 200
    assert response.json() == {"summary": "Mock Model Summary"}
    mock_predictor.get_model_summary_string.assert_called_once()

def test_get_model_summary_no_model(mock_predictor):
    """Test retrieving model summary when no model is loaded."""
    mock_predictor.model = None
    mock_predictor.get_model_summary_string.return_value = "Model not loaded or built yet."
    response = client.get("/model-summary")
    assert response.status_code == 400
    assert "Model not loaded or built yet" in response.json()["detail"]

def test_get_training_progress_empty():
    """Test retrieving training progress when file is empty or not started."""
    # Ensure the history file is initially empty by fixture
    response = client.get("/training-progress")
    assert response.status_code == 200
    assert response.json() == {'epochs': [], 'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

def test_get_training_progress_with_data(mock_data_loader):
    """Test retrieving training progress with some data."""
    # Populate the dummy history file
    history_data = {
        'epochs': [1, 2],
        'loss': [0.5, 0.4],
        'accuracy': [0.6, 0.7],
        'val_loss': [0.6, 0.5],
        'val_accuracy': [0.55, 0.65]
    }
    with open(os.path.join(mock_data_loader.data_dir, "current_training_history.json"), 'w') as f:
        json.dump(history_data, f)

    response = client.get("/training-progress")
    assert response.status_code == 200
    assert response.json() == history_data

def test_dashboard_model_loaded_status(mock_predictor):
    """Test dashboard correctly reflects model loaded status."""
    # Case 1: Model is loaded
    mock_predictor.model = MagicMock()
    response = client.get("/dashboard")
    assert response.status_code == 200
    # Assert Jinja2 context variable is passed correctly
    assert '"model_loaded_status": true' in response.text
    assert '"total_params": 100000' in response.text # From mock_predictor fixture

    # Case 2: Model is not loaded
    mock_predictor.model = None
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert '"model_loaded_status": false' in response.text
    assert '"total_params": "N/A"' in response.text # Should be N/A when not loaded

