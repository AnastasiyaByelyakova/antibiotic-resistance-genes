from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Depends, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.exceptions import RequestValidationError # Import this for specific validation errors


import uvicorn
import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import traceback
import shutil
import tensorflow as tf # Import tensorflow to get version

# Imports for authentication
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Import database operations and dotenv
from dotenv import load_dotenv
from database import (
    get_model_metadata, update_model_metadata,
    insert_training_run, insert_retraining_run, insert_validation_run,
    insert_prediction, get_recent_predictions, get_recent_validations,
    increment_total_predictions_and_update_last_time,
    get_user_by_username, create_user, verify_password
)

# Load environment variables
load_dotenv()

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
templates.env.filters["date"] = lambda dt, fmt: dt.strftime(fmt) if isinstance(dt, datetime) else dt

# Initialize components (will be fully initialized in startup_event)
data_loader = None # Will be initialized as DataLoader
predictor = None # Will be initialized as ResistancePredictor

# Global variable to hold the path to the current training history file
current_training_history_file: str = ""

# Ensure directories exist
os.makedirs("uploads", exist_ok=True) # For temporary uploaded files
os.makedirs("data", exist_ok=True)    # For metadata, label_encoder, etc.
os.makedirs("models", exist_ok=True)  # For trained model files
os.makedirs("templates", exist_ok=True) # Ensure templates directory exists

# --- Authentication Configuration ---
# Generate a strong secret key for JWT in your .env file
# openssl rand -hex 32
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256" # Or "HS384", "HS512"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 # Token expiration time

if not SECRET_KEY:
    logger.error("SECRET_KEY environment variable is not set. JWT authentication will not work.")
    raise ValueError("SECRET_KEY environment variable is not set. Please set it in your .env file.")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Custom OAuth2PasswordBearer that checks cookies first, then headers
class OAuth2PasswordBearerWithCookie(OAuth2PasswordBearer):
    async def __call__(self, request: Request) -> Optional[str]:
        # Check for token in cookie first
        token: Optional[str] = request.cookies.get("access_token")
        if token and token.startswith("Bearer "):
            return token.replace("Bearer ", "")
        
        # If not in cookie, fall back to checking Authorization header
        try:
            # Call the parent's __call__ method to check the Authorization header
            header_token = await super().__call__(request)
            return header_token
        except HTTPException:
            # If the header also doesn't contain a valid token,
            # this will eventually lead to an HTTPException from get_current_active_user
            return None

oauth2_scheme = OAuth2PasswordBearerWithCookie(tokenUrl="token")


# Pydantic models for request/response bodies
class User(BaseModel):
    username: str

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# --- Authentication Helper Functions ---

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """Helper to decode and validate JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return get_user_by_username(username)
    except JWTError:
        return None

async def get_current_active_user(token: Optional[str] = Depends(oauth2_scheme)):
    """Dependency to get current authenticated user, raising HTTPException on failure."""
    if token is None:
        # If no token is found by OAuth2PasswordBearerWithCookie,
        # raise HTTPException to trigger the global exception handler
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await get_current_user_from_token(token)
    if user is None:
        # Token found but invalid (e.g., expired, tampered).
        # Raise HTTPException to trigger the global exception handler
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# --- Global Exception Handler for 401 Unauthorized ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # If the exception is 401 Unauthorized
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        # Check if the request is for an HTML page (browser navigation)
        # We assume if "text/html" is in Accept header, it's a browser request wanting HTML
        # Also, for GET requests, we generally want HTML redirects.
        if "text/html" in request.headers.get("accept", "") or request.method == "GET":
            # For HTML requests, redirect to login page
            logger.warning(f"Unauthorized access to {request.url.path}. Redirecting to login.")
            return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
        else:
            # For API requests (e.g., AJAX, curl), return JSON error
            logger.warning(f"Unauthorized API access to {request.url.path}. Returning JSON error.")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": exc.detail},
                headers=exc.headers
            )
    # For all other HTTPExceptions, use FastAPI's default handler or re-raise
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers
    )


# --- Authentication Endpoints ---

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login page."""
    # If user is already logged in, redirect to dashboard
    access_token_cookie = request.cookies.get("access_token")
    if access_token_cookie:
        token_string = access_token_cookie.replace("Bearer ", "")
        user = await get_current_user_from_token(token_string)
        if user:
            # User is authenticated, redirect to main app page
            return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/token", response_model=Token)
async def login_for_access_token(response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    """Endpoint for user login to obtain an access token."""
    user_db = get_user_by_username(form_data.username)
    if not user_db or not verify_password(form_data.password, user_db['hashed_password']):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_db['username']}, expires_delta=access_token_expires
    )
    
    # Set the token as an HTTP-only cookie
    response.set_cookie(
        key="access_token", 
        value=f"Bearer {access_token}", # Store with "Bearer " prefix for clarity
        httponly=True, 
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60, # max_age in seconds
        expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/"
    )
    # Return JSON for API clients. Browser will handle redirection via JS on success.
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Serve the registration page."""
    # If user is already logged in, redirect to dashboard
    access_token_cookie = request.cookies.get("access_token")
    if access_token_cookie:
        token_string = access_token_cookie.replace("Bearer ", "")
        user = await get_current_user_from_token(token_string)
        if user:
            return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register_user(request: Request, username: str = Form(...), password: str = Form(...)):
    """Endpoint for user registration."""
    try:
        user_id = create_user(username, password)
        if user_id:
            return JSONResponse(status_code=status.HTTP_201_CREATED, content={"message": "User registered successfully!"})
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to register user.")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error during registration: {traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred during registration.")

@app.get("/logout")
async def logout(response: Response):
    """Endpoint to log out by clearing the access token cookie."""
    response.delete_cookie(key="access_token", path="/")
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


# --- Application Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Initialize the application components and load existing model."""
    logger.info("Starting Bacterial Drug Resistance Predictor API...")
    
    # Imports inside startup event to avoid circular dependencies during initial import if DataLoader/Predictor need DB
    from data_loader import DataLoader
    from model import ResistancePredictor

    global data_loader, predictor
    data_loader = DataLoader()
    predictor = ResistancePredictor(data_loader_instance=data_loader)    
    
    # Try to load existing model
    try:
        predictor.load_model()
        logger.info("Existing model loaded successfully at startup.")
        # Update model_metadata in DB if model was loaded
        # Fetch current metadata to avoid overwriting values not related to model loading
        db_metadata = get_model_metadata()
        update_model_metadata(
            model_loaded=True,
            input_sequence_length=predictor.model.input_shape[1] if predictor.model.input_shape else None,
            antibiotic_labels_count=predictor.model.output_shape[1] if predictor.model.output_shape else None,
            total_params=predictor.model.count_params(),
            layers_count=len(predictor.model.layers),
            keras_version=tf.__version__,
            tensorflow_version=tf.__version__,
            current_model_path=os.path.join(predictor.model_dir, 'resistance_predictor_model.keras'),
            label_encoder_path=os.path.join(data_loader.data_dir, 'label_encoder.pkl'),
            total_predictions_made=db_metadata.get('total_predictions_made', 0), # Preserve existing counts
            unique_files_predicted=db_metadata.get('unique_files_predicted', 0), # Preserve existing counts
            last_prediction_time=db_metadata.get('last_prediction_time', None),
            frequent_resistances=db_metadata.get('frequent_resistances', {})
        )
    except Exception as e:
        logger.warning(f"No existing model found or failed to load: {e}. Model needs to be trained.")
        # Ensure metadata reflects no model loaded
        update_model_metadata(model_loaded=False)

# Updated HTML-serving routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Serves the main prediction page, requiring authentication."""
    # If get_current_active_user raises HTTPException, the global handler will redirect
    # If it succeeds, current_user is available
    return templates.TemplateResponse("index_template.html", {"request": request, "username": current_user['username']})


@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request, current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Serves the dashboard page, requiring authentication."""
    # If get_current_active_user raises HTTPException, the global handler will redirect
    # If it succeeds, current_user is available
    logger.info(f"Dashboard accessed by user: {current_user['username']}")
    metadata = get_model_metadata()
    metadata.setdefault("model_loaded_status", False)
    metadata.setdefault("input_sequence_length", "N/A")
    metadata.setdefault("antibiotic_labels_count", "N/A")
    metadata.setdefault("total_params", "N/A")
    metadata.setdefault("layers_count", "N/A")
    metadata.setdefault("keras_version", tf.__version__)
    metadata.setdefault("tensorflow_version", tf.__version__)
    metadata.setdefault("train_samples", "N/A")
    metadata.setdefault("test_samples", "N/A")
    metadata.setdefault("test_accuracy", "N/A")
    metadata.setdefault("last_trained", "N/A")
    metadata.setdefault("total_predictions", metadata.get("total_predictions_made", 0))
    metadata.setdefault("unique_files_predicted", metadata.get("unique_files_predicted", 0))
    metadata.setdefault("last_prediction_time", metadata.get("last_prediction_time", "N/A"))
    metadata.setdefault("frequent_resistances", metadata.get("frequent_resistances", {}))

    recent_predictions_db = get_recent_predictions()
    recent_validations_db = get_recent_validations()

    return templates.TemplateResponse("dashboard_template.html", {
        "request": request,
        "metadata": metadata,
        "history": recent_predictions_db,
        "validations": recent_validations_db,
        "now": datetime.now(),
        "username": current_user['username']
    })


# --- Application Endpoints (API routes protected by Depends) ---
# These routes will also now use the global exception handler for 401.

@app.post("/predict")
async def predict_resistance(request: Request, file: UploadFile = File(...), current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """
    Predicts antibiotic resistance from an uploaded FASTA file and saves results to DB.
    Requires authentication.
    """
    logger.info(f"Received prediction request for file: {file.filename} from user: {current_user['username']}")
    temp_fasta_path = ""
    try:
        if predictor.model is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model is not trained/loaded. Please train a model first.")

        temp_fasta_path = os.path.join("uploads", file.filename)
        with open(temp_fasta_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded FASTA saved to {temp_fasta_path}")

        predictions_data = predictor.predict_from_fasta(temp_fasta_path)
        
        for seq_id, preds in predictions_data.items():
            predicted_drugs = [drug for drug, prob in preds.items() if prob >= 0.5]
            insert_prediction(
                input_filename=file.filename,
                sequence_id=seq_id,
                predicted_resistances=predicted_drugs,
                probabilities=preds
            )
        
        current_meta = get_model_metadata()
        current_frequent_resistances = current_meta.get('frequent_resistances', {})
        for seq_id, preds in predictions_data.items():
            for drug_class, prob in preds.items():
                if prob >= 0.5:
                    current_frequent_resistances[drug_class] = current_frequent_resistances.get(drug_class, 0) + 1
        
        update_model_metadata(
            frequent_resistances=current_frequent_resistances,
            last_prediction_time=datetime.now().isoformat()
        )
        increment_total_predictions_and_update_last_time(file.filename)


        return JSONResponse(content=predictions_data)
    except HTTPException as e:
        logger.error(f"Prediction HTTP Error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error during prediction: {traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred: {e}")
    finally:
        if os.path.exists(temp_fasta_path):
            os.remove(temp_fasta_path)

@app.get("/model-summary", response_class=JSONResponse)
async def get_model_summary(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """
    Returns the Keras model summary as a string.
    Requires authentication.
    """
    if predictor and predictor.model:
        model_summary_str = predictor.get_model_summary_string()
        return JSONResponse(content={"summary": model_summary_str})
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model not loaded or built yet. Train a model first.")

@app.get("/training-progress", response_class=JSONResponse)
async def get_training_progress(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """
    Returns the current training progress from the log file.
    This remains file-based for real-time chart updates during active training.
    Requires authentication.
    """
    if not current_training_history_file or not os.path.exists(current_training_history_file):
        return JSONResponse(content={'epochs': [], 'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []})
    
    try:
        with open(current_training_history_file, 'r') as f:
            history_data = json.load(f)
        return JSONResponse(content=history_data)
    except Exception as e:
        logger.error(f"Error reading training history file: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not retrieve training progress.")

@app.post("/train")
async def train_model(
    request: Request,
    file: UploadFile = File(..., description="ZIP file containing training data"),
    epochs: int = Form(50, description="Number of epochs for initial training"),
    test_size: float = Form(0.2, description="Fraction of data for testing (0.0 to 1.0)"),
    current_user: Dict[str, Any] = Depends(get_current_active_user) # Protect this endpoint
):
    """
    Initializes and trains a new model from scratch.
    Requires a ZIP file with FASTA sequences and a metadata.json.
    Saves training details to MySQL. Requires authentication.
    """
    logger.info(f"Received request to train model by user: {current_user['username']}. File: {file.filename}, Epochs: {epochs}, Test Size: {test_size}")
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

        current_training_history_file = os.path.join(data_loader.data_dir, 'current_training_history.json')
        
        training_info = predictor.train(X_train, y_train, X_test, y_test, epochs=epochs, training_history_filepath=current_training_history_file)

        insert_training_run(
            filename=file.filename,
            epochs_trained=training_info.get('epochs_trained', 0),
            train_samples=len(X_train),
            test_samples=len(X_test),
            final_val_loss=training_info.get('final_val_loss', None),
            final_val_accuracy=training_info.get('final_val_accuracy', None),
            test_loss=training_info.get('test_loss', None),
            test_accuracy=training_info.get('test_accuracy', None),
            hamming_loss=training_info.get('hamming_loss', None),
            jaccard_score=training_info.get('jaccard_score', None),
            model_version=training_info.get('model_version', '1.0'),
            keras_version=tf.__version__,
            tensorflow_version=tf.__version__,
            classification_report=training_info.get('classification_report', {})
        )

        update_model_metadata(
            model_loaded=True,
            input_sequence_length=X_train.shape[1] if X_train.shape else None,
            antibiotic_labels_count=y_train.shape[1] if y_train.shape else None,
            total_params=predictor.model.count_params() if predictor.model else None,
            layers_count=len(predictor.model.layers) if predictor.model else None,
            keras_version=tf.__version__,
            tensorflow_version=tf.__version__,
            current_model_path=os.path.join(predictor.model_dir, 'resistance_predictor_model.keras'),
            label_encoder_path=os.path.join(data_loader.data_dir, 'label_encoder.pkl'),
            train_samples=len(X_train),
            test_samples=len(X_test),
            test_accuracy=training_info.get('test_accuracy', None),
            last_trained=training_info.get('timestamp', datetime.now().isoformat())
        )

        return JSONResponse(content={"message": "Model training initiated successfully! Model is now ready.", "training_info": training_info})

    except ValueError as e:
        logger.error(f"Training data error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error during model training: {traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred during training: {e}. Check server logs for details.")
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

@app.post("/retrain")
async def retrain_model(
    request: Request,
    file: UploadFile = File(..., description="ZIP file containing new training data"),
    epochs: int = Form(10, description="Number of epochs to retrain the model"),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Retrains an existing model with additional data and saves details to DB.
    Requires authentication.
    """
    logger.info(f"Received request to retrain model by user: {current_user['username']}. File: {file.filename}, Epochs: {epochs}")
    temp_zip_path = ""
    try:
        if predictor.model is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No model loaded. Please train a model first before retraining.")

        temp_zip_path = os.path.join("uploads", file.filename)
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded retraining ZIP saved to {temp_zip_path}")

        X_new, X_x, y_new, yy, zip_metadata = data_loader.load_data_from_uploaded_zip(
            temp_zip_path, is_training_data=True
        )
        if X_new is None or y_new is None:
            raise ValueError("Failed to load and preprocess new data from ZIP for retraining. Check ZIP content and format.")

        retrain_info = predictor.retrain(X_new, y_new, epochs=epochs)

        insert_retraining_run(
            filename=file.filename,
            new_samples=retrain_info.get('new_samples', 0),
            epochs_retrained=retrain_info.get('epochs_retrained', 0),
            final_loss=retrain_info.get('final_loss', None),
            model_version=retrain_info.get('model_version', '1.0')
        )

        update_model_metadata(
            model_loaded=True,
            last_trained=retrain_info.get('timestamp', datetime.now().isoformat())
        )

        return JSONResponse(content={"message": "Model retraining completed successfully!", "retrain_info": retrain_info})

    except HTTPException as e:
        logger.error(f"Retraining HTTP Error: {e.detail}")
        raise e
    except ValueError as e:
        logger.error(f"Retraining data error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error during model retraining: {traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred during retraining: {e}. Check server logs for details.")
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

@app.post("/validate")
async def validate_model(
    request: Request,
    file: UploadFile = File(..., description="ZIP file containing validation data"),
    validation_name: str = Form(..., description="Name for this validation run"),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Validates the currently loaded model using a provided dataset and saves details to DB.
    Requires authentication.
    """
    logger.info(f"Received request to validate model by user: {current_user['username']}. File: {file.filename}, Name: {validation_name}")
    temp_zip_path = ""
    try:
        if predictor.model is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No model loaded. Please train a model first before validation.")

        temp_zip_path = os.path.join("uploads", file.filename)
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded validation ZIP saved to {temp_zip_path}")

        X_val, X_x, y_val, yy, zip_metadata = data_loader.load_data_from_uploaded_zip(
            temp_zip_path, is_training_data=False
        )
        if X_val is None or y_val is None:
            raise ValueError("Failed to load and preprocess validation data from ZIP. Check ZIP content and format.")

        validation_metrics = predictor.evaluate_model(X_val, y_val)
        
        insert_validation_run(
            validation_name=validation_name,
            filename=file.filename,
            num_samples=len(X_val),
            accuracy=validation_metrics.get('accuracy', None),
            f1_macro=validation_metrics.get('f1_macro', None),
            hamming_loss=validation_metrics.get('hamming_loss', None),
            report=json.loads(validation_metrics.get('report', '{}'))
        )

        return JSONResponse(content={"message": "Model validation completed successfully!", "validation_results": validation_metrics})

    except HTTPException as e:
        logger.error(f"Validation HTTP Error: {e.detail}")
        raise e
    except ValueError as e:
        logger.error(f"Validation data error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error during model validation: {traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=f"An internal error occurred during validation: {e}. Check server logs for details.")
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

