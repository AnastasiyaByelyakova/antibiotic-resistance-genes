from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
import traceback
import shutil
import tensorflow as tf # Import tensorflow to get version
from Bio import SeqIO 
from data_loader import DataLoader
from model import ResistancePredictor 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bacterial Drug Resistance Predictor",
    description="ML Pipeline for predicting antibiotic resistance from bacterial genomes",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")

# Add a custom Jinja2 filter for date formatting immediately after initialization
templates.env.filters["date"] = lambda dt, fmt: dt.strftime(fmt)

# Initialize components (will be fully initialized in startup_event)
data_loader: Optional[DataLoader] = None
predictor: Optional[ResistancePredictor] = None

# Global variables for storing recent results and current training progress
recent_validations = []
MAX_RECENT_VALIDATIONS = 10
recent_predictions = [] # Assuming this exists for dashboard history
MAX_RECENT_PREDICTIONS = 20 # Max predictions to store in memory

# Global variable to hold the path to the current training history file
current_training_history_file: str = ""

# Ensure directories exist
os.makedirs("uploads", exist_ok=True) # For temporary uploaded files
os.makedirs("data", exist_ok=True)    # For metadata, label_encoder, etc.
os.makedirs("models", exist_ok=True)  # For trained model files

@app.on_event("startup")
async def startup_event():
    """Initialize the application components and load existing model."""
    logger.info("Starting Bacterial Drug Resistance Predictor API...")
    
    global data_loader, predictor
    data_loader = DataLoader()
    predictor = ResistancePredictor(data_loader_instance=data_loader)    
    # Try to load existing model
    try:
        predictor.load_model()
        logger.info("Existing model loaded successfully at startup.")
    except Exception as e:
        logger.warning(f"No existing model found or failed to load: {e}. Model needs to be trained.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main prediction page."""
    return templates.TemplateResponse("index_template.html", {"request": request})

@app.post("/predict")
async def predict_resistance(request: Request, file: UploadFile = File(...)):
    """
    Predicts antibiotic resistance from an uploaded FASTA file.
    """
    logger.info(f"Received prediction request for file: {file.filename}")
    temp_fasta_path = ""
    try:
        if predictor.model is None:
            raise HTTPException(status_code=400, detail="Model is not trained/loaded. Please train a model first.")

        # Save the uploaded file temporarily
        temp_fasta_path = os.path.join("uploads", file.filename)
        with open(temp_fasta_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded FASTA saved to {temp_fasta_path}")

        # Predict
        predictions_data = predictor.predict_from_fasta(temp_fasta_path)
        
        # Update dashboard metadata
        try:
            with open(os.path.join(data_loader.data_dir, 'metadata.json'), 'r') as f:
                current_metadata = json.load(f)
        except FileNotFoundError:
            current_metadata = {}
        
        current_metadata["total_predictions"] = current_metadata.get("total_predictions", 0) + 1
        current_metadata["last_prediction_time"] = datetime.now().isoformat()
        
        unique_files = set(current_metadata.get("unique_files_predicted_list", []))
        unique_files.add(file.filename)
        current_metadata["unique_files_predicted_list"] = list(unique_files)
        current_metadata["unique_files_predicted"] = len(unique_files)

        # Update frequent resistances
        frequent_resistances = current_metadata.get("frequent_resistances", {})
        for seq_id, preds in predictions_data.items():
            for drug_class, prob in preds.items():
                if prob >= 0.5: # Assuming 0.5 as threshold for "predicted" resistance
                    frequent_resistances[drug_class] = frequent_resistances.get(drug_class, 0) + 1
        current_metadata["frequent_resistances"] = frequent_resistances

        with open(os.path.join(data_loader.data_dir, 'metadata.json'), 'w') as f:
            json.dump(current_metadata, f, indent=2)

        # Add to recent predictions for dashboard
        for seq_id, preds in predictions_data.items():
            predicted_drugs = [drug for drug, prob in preds.items() if prob >= 0.5]
            recent_predictions.insert(0, {
                "filename": file.filename,
                "sequence_id": seq_id,
                "timestamp": datetime.now().isoformat(),
                "predicted_resistances": predicted_drugs,
                "total_resistances": len(predicted_drugs)
            })
            if len(recent_predictions) > MAX_RECENT_PREDICTIONS:
                recent_predictions.pop()
        return JSONResponse(content=predictions_data)
    except HTTPException as e:
        logger.error(f"Prediction HTTP Error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error during prediction: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        if os.path.exists(temp_fasta_path):
            os.remove(temp_fasta_path) # Clean up temp file

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Serve the dashboard page with model and prediction statistics."""
    metadata = {}
    try:
        with open(os.path.join("data", "metadata.json"), "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        pass # Metadata will be empty if not found

    # Ensure model_loaded_status is accurate based on actual predictor state
    metadata["model_loaded_status"] = predictor is not None and predictor.model is not None

    # Dynamically get model info for dashboard
    if predictor and predictor.model:
        metadata["input_sequence_length"] = predictor.model.input_shape[1] if predictor.model.input_shape else 'N/A'
        metadata["antibiotic_labels_count"] = predictor.model.output_shape[1] if predictor.model.output_shape else 'N/A'
        metadata["total_params"] = predictor.model.count_params()
        metadata["layers_count"] = len(predictor.model.layers)
        metadata["keras_version"] = tf.__version__
        metadata["tensorflow_version"] = tf.__version__
    else:
        # If model not loaded, ensure these are N/A
        metadata.update({
            "input_sequence_length": "N/A",
            "antibiotic_labels_count": "N/A",
            "total_params": "N/A",
            "layers_count": "N/A",
            "keras_version": tf.__version__, # Still show installed version
            "tensorflow_version": tf.__version__,
        })

    return templates.TemplateResponse("dashboard_template.html", {
        "request": request,
        "metadata": metadata,
        "history": recent_predictions, # Pass recent predictions history
        "validations": recent_validations, # Pass recent validation history
        "now": datetime.now() # Pass the current datetime object
    })

@app.get("/model-summary", response_class=JSONResponse)
async def get_model_summary():
    """
    Returns the Keras model summary as a string.
    """
    if predictor and predictor.model:
        model_summary_str = predictor.get_model_summary_string()
        return JSONResponse(content={"summary": model_summary_str})
    else:
        raise HTTPException(status_code=400, detail="Model not loaded or built yet. Train a model first.")

@app.get("/training-progress", response_class=JSONResponse)
async def get_training_progress():
    """
    Returns the current training progress from the log file.
    """
    if not current_training_history_file or not os.path.exists(current_training_history_file):
        # Return an empty but structured response if no training is active or file not found
        return JSONResponse(content={'epochs': [], 'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []})
    
    try:
        with open(current_training_history_file, 'r') as f:
            history_data = json.load(f)
        return JSONResponse(content=history_data)
    except Exception as e:
        logger.error(f"Error reading training history file: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve training progress.")

@app.post("/train")
async def train_model(
    request: Request,
    file: UploadFile = File(..., description="ZIP file containing training data"),
    epochs: int = Form(50, description="Number of epochs for initial training"),
    test_size: float = Form(0.2, description="Fraction of data for testing (0.0 to 1.0)")
):
    """
    Initializes and trains a new model from scratch.
    Requires a ZIP file with FASTA sequences and a metadata.json.
    """
    logger.info(f"Received request to train model. File: {file.filename}, Epochs: {epochs}, Test Size: {test_size}")
    temp_zip_path = ""
    global current_training_history_file # Declare as global to modify it
    try:
        # Save the uploaded ZIP file temporarily
        temp_zip_path = os.path.join("uploads", file.filename)
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded training ZIP saved to {temp_zip_path}")

        X_train, X_test, y_train, y_test, metadata_from_zip = data_loader.load_data_from_uploaded_zip(
            temp_zip_path, is_training_data=True, test_size=test_size
        )

        if X_train is None or X_test is None or y_train is None or y_test is None:
            raise ValueError("Failed to load and preprocess data from ZIP. Check ZIP content and format.")

        # Define the path for the training history log file
        current_training_history_file = os.path.join(data_loader.data_dir, 'current_training_history.json')
        
        # Train the model, passing the history file path
        training_info = predictor.train(X_train, y_train, X_test, y_test, epochs=epochs, training_history_filepath=current_training_history_file)

        # Update global metadata after training
        metadata = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_accuracy": training_info.get('test_accuracy', 'N/A'),
            "last_trained": training_info.get('timestamp', datetime.now().isoformat()),
            "model_version": training_info.get('model_version', '1.0'),
            "input_sequence_length": X_train.shape[1] if X_train.shape else 'N/A',
            "antibiotic_labels_count": y_train.shape[1] if y_train.shape else 'N/A',
            "model_loaded_status": True,
            "total_params": training_info.get('total_params', 'N/A'),
            "layers_count": training_info.get('layers_count', 'N/A'),
            "keras_version": tf.__version__,
            "tensorflow_version": tf.__version__,
        }
        # Save this to data/metadata.json
        with open(os.path.join(data_loader.data_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        return JSONResponse(content={"message": "Model training initiated successfully! Model is now ready.", "training_info": training_info})

    except ValueError as e:
        logger.error(f"Training data error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during model training: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during training: {e}. Check server logs for details.")
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path) # Clean up temp file

@app.post("/retrain")
async def retrain_model(
    request: Request,
    file: UploadFile = File(..., description="ZIP file containing new training data"),
    epochs: int = Form(10, description="Number of epochs to retrain the model")
):
    """
    Retrains an existing model with additional data.
    Requires a ZIP file with FASTA sequences and a metadata.json.
    """
    logger.info(f"Received request to retrain model. File: {file.filename}, Epochs: {epochs}")
    temp_zip_path = ""
    try:
        if predictor.model is None:
            raise HTTPException(status_code=400, detail="No model loaded. Please train a model first before retraining.")

        temp_zip_path = os.path.join("uploads", file.filename)
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded retraining ZIP saved to {temp_zip_path}")

        X_new, X_x, y_new, yy, zip_metadata = data_loader.load_data_from_uploaded_zip(
            temp_zip_path, is_training_data=True
        )
        if X_new is None or y_new is None:
            raise ValueError("Failed to load and preprocess new data from ZIP for retraining. Check ZIP content and format.")

        # Retrain the model
        retrain_info = predictor.retrain(X_new, y_new, epochs=epochs)

        # Update metadata after retraining (e.g., last_trained time)
        try:
            with open(os.path.join(data_loader.data_dir, 'metadata.json'), 'r') as f:
                current_metadata = json.load(f)
        except FileNotFoundError:
            current_metadata = {}

        current_metadata["last_trained"] = retrain_info.get('timestamp', datetime.now().isoformat())
        with open(os.path.join(data_loader.data_dir, 'metadata.json'), 'w') as f:
            json.dump(current_metadata, f, indent=2)

        return JSONResponse(content={"message": "Model retraining completed successfully!", "retrain_info": retrain_info})

    except HTTPException as e:
        logger.error(f"Retraining HTTP Error: {e.detail}")
        raise e
    except ValueError as e:
        logger.error(f"Retraining data error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during model retraining: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during retraining: {e}. Check server logs for details.")
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path) # Clean up temp file

@app.post("/validate")
async def validate_model(
    request: Request,
    file: UploadFile = File(..., description="ZIP file containing validation data"),
    validation_name: str = Form(..., description="Name for this validation run")
):
        """
        Validates the currently loaded model using a provided dataset.
        Requires a ZIP file with FASTA sequences and a metadata.json.
        """
        logger.info(f"Received request to validate model. File: {file.filename}, Name: {validation_name}")
        temp_zip_path = ""
        try:
            if predictor.model is None:
                raise HTTPException(status_code=400, detail="No model loaded. Please train a model first before validation.")

            temp_zip_path = os.path.join("uploads", file.filename)
            with open(temp_zip_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Uploaded validation ZIP saved to {temp_zip_path}")

            X_val, X_x, y_val, yy, zip_metadata = data_loader.load_data_from_uploaded_zip(
                temp_zip_path, is_training_data=True
            )
            if X_val is None or y_val is None:
                raise ValueError("Failed to load and preprocess validation data from ZIP. Check ZIP content and format.")

            validation_metrics = predictor.evaluate_model(X_val, y_val)
            validation_entry = {
                "name": validation_name,
                "filename": file.filename,
                "timestamp": datetime.now().isoformat(),
                "num_samples": len(X_val),
                "metrics": {
                    "accuracy": validation_metrics.get('accuracy', 'N/A'),
                    "f1_macro": validation_metrics.get('f1_macro', 'N/A'),
                    "hamming_loss": validation_metrics.get('hamming_loss', 'N/A'),
                },
                "report": validation_metrics.get('report', 'No classification report available.')
            }
            recent_validations.insert(0, validation_entry)
            if len(recent_validations) > MAX_RECENT_VALIDATIONS:
                recent_validations.pop()

            return JSONResponse(content={"message": "Model validation completed successfully!", "validation_results": validation_entry})

        except HTTPException as e:
            logger.error(f"Validation HTTP Error: {e.detail}")
            raise e
        except ValueError as e:
            logger.error(f"Validation data error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error during model validation: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"An internal error occurred during validation: {e}. Check server logs for details.")
        finally:
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

