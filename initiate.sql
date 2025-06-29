-- Create the database if it doesn't exist
 CREATE DATABASE IF NOT EXISTS bdr_predictor_db;

-- Use the created database
 USE bdr_predictor_db;

-- Table to store details of model training runs
CREATE TABLE IF NOT EXISTS training_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    filename VARCHAR(255) NOT NULL,
    epochs_trained INT NOT NULL,
    train_samples INT,
    test_samples INT,
    final_val_loss FLOAT,
    final_val_accuracy FLOAT,
    test_loss FLOAT,
    test_accuracy FLOAT,
    hamming_loss FLOAT,
    jaccard_score FLOAT,
    model_version VARCHAR(50),
    keras_version VARCHAR(50),
    tensorflow_version VARCHAR(50),
    classification_report JSON, -- Store as JSON string
    full_report_path VARCHAR(255) -- Optional: path to a larger report file
);

-- Table to store details of model retraining runs
CREATE TABLE IF NOT EXISTS retraining_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    filename VARCHAR(255) NOT NULL,
    new_samples INT,
    epochs_retrained INT NOT NULL,
    final_loss FLOAT,
    model_version VARCHAR(50) -- Version of the model after retraining
);

-- Table to store details of model validation runs
CREATE TABLE IF NOT EXISTS validation_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    validation_name VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    num_samples INT,
    accuracy FLOAT,
    f1_macro FLOAT,
    hamming_loss FLOAT,
    report JSON -- Store classification report as JSON string
);

-- Table to store individual predictions
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    input_filename VARCHAR(255) NOT NULL,
    sequence_id VARCHAR(255) NOT NULL,
    predicted_resistances JSON NOT NULL, -- Store list of predicted drugs as JSON array
    probabilities JSON NOT NULL -- Store probabilities as JSON object {drug: prob, ...}
);

-- Table to store general model metadata (e.g., last trained, current model path, summary info)
-- This table will typically have only one or a few rows, representing the *current* state.
CREATE TABLE IF NOT EXISTS model_metadata (
    id INT AUTO_INCREMENT PRIMARY KEY,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    model_loaded BOOLEAN DEFAULT FALSE,
    current_model_path VARCHAR(255),
    label_encoder_path VARCHAR(255),
    input_sequence_length INT,
    antibiotic_labels_count INT,
    total_params BIGINT,
    layers_count INT,
    keras_version VARCHAR(50),
    tensorflow_version VARCHAR(50),
    total_predictions_made BIGINT DEFAULT 0,
    unique_files_predicted BIGINT DEFAULT 0,
    last_prediction_time DATETIME,
    frequent_resistances JSON -- Store as JSON object {drug: count, ...}
);

-- Index for faster lookup on predictions
CREATE INDEX idx_predictions_input_filename ON predictions (input_filename);
CREATE INDEX idx_predictions_timestamp ON predictions (timestamp);

-- Index for faster lookup on validation runs
CREATE INDEX idx_validation_runs_timestamp ON validation_runs (timestamp);

-- Add unique constraint to model_metadata if you want to ensure only one global entry
-- ALTER TABLE model_metadata ADD CONSTRAINT uc_model_metadata UNIQUE (id);
ALTER TABLE model_metadata
    ADD COLUMN train_samples INT,
    ADD COLUMN test_samples INT;

ALTER TABLE model_metadata
ADD COLUMN test_accuracy FLOAT;

-- Table to store user credentials
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
