import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import json
import os
import joblib
from datetime import datetime
import logging
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import classification_report, hamming_loss, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
import io # Import io to capture model summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingHistoryLogger(tf.keras.callbacks.Callback):
    """
    A custom Keras callback to log training history (loss, accuracy, val_loss, val_accuracy)
    to a JSON file after each epoch. This allows for real-time monitoring.
    """
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.history_data = {'epochs': [], 'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        # Ensure the file is clear at the start of training
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        with open(self.filepath, 'w') as f:
            json.dump(self.history_data, f) # Initialize with empty structure

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        logs = logs or {}
        self.history_data['epochs'].append(epoch + 1)
        self.history_data['loss'].append(logs.get('loss'))
        self.history_data['accuracy'].append(logs.get('accuracy'))
        self.history_data['val_loss'].append(logs.get('val_loss'))
        self.history_data['val_accuracy'].append(logs.get('val_accuracy'))

        # Write current history to file
        with open(self.filepath, 'w') as f:
            json.dump(self.history_data, f, indent=2)
        logger.info(f"Epoch {epoch+1} history logged to {self.filepath}")


class ResistancePredictor:
    def __init__(self, model_dir: str = "models", data_dir: str = "data", data_loader_instance=None):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None
        self.history = None
        self.metadata = None
        self.label_encoder = None
        self.data_loader = data_loader_instance
        
        os.makedirs(model_dir, exist_ok=True)
    
    def build_model(self, input_dim: int, n_classes: int) -> Model:
        """Build CNN-LSTM model for sequence classification"""
        
        # Input layer
        inputs = layers.Input(shape=(input_dim,), dtype=tf.float32)

        x = layers.Reshape((input_dim, 1))(inputs) # Reshape for Conv1D

        # Convolutional Block
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)

        # LSTM Layer
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x) # return_sequences=False for classification
        x = layers.Dropout(0.5)(x)

        # Output layer
        outputs = layers.Dense(n_classes, activation='sigmoid')(x) # Sigmoid for multi-label classification

        model = Model(inputs=inputs, outputs=outputs, name="cnn_lstm_resistance_predictor")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Capture model summary and log it
        model_summary_str = self.get_model_summary_string(model)
        logger.info(f"Model Summary:\n{model_summary_str}")
        
        return model

    def get_model_summary_string(self, model_instance: Optional[Model] = None) -> str:
        """
        Returns the Keras model summary as a string.
        If no model_instance is provided, uses self.model.
        """
        model_to_summarize = model_instance if model_instance else self.model
        if model_to_summarize is None:
            return "Model not loaded or built yet."
        
        # Use StringIO to capture the summary printed to stdout
        string_list = []
        model_to_summarize.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 50, mlb: Optional[MultiLabelBinarizer] = None, training_history_filepath: Optional[str] = None):
        """Train the model with given data."""
        if mlb is not None:
            self.label_encoder = mlb
            logger.info("Using provided MultiLabelBinarizer instance for training.")
        else:
            try:
                self.label_encoder = joblib.load(os.path.join(self.data_dir, 'label_encoder.pkl'))
                logger.info("MultiLabelBinarizer loaded from disk for training.")
            except FileNotFoundError:
                logger.error("MultiLabelBinarizer not found. Cannot perform classification report.")
                raise RuntimeError("Label encoder not found. Ensure DataLoader has saved it during data preprocessing.")

        input_dim = X_train.shape[1]
        n_classes = y_train.shape[1]

        # Rebuild model if it's not initialized, or if input_dim or n_classes changed
        if (self.model is None or
            (self.model.input_shape and self.model.input_shape[1] != input_dim) or # Check input_dim
            (self.model.output_shape and self.model.output_shape[1] != n_classes)):
            logger.info(f"Building/rebuilding model due to change in input_dim ({self.model.input_shape[1] if self.model and self.model.input_shape else 'None'} -> {input_dim}) or n_classes.")
            self.model = self.build_model(input_dim, n_classes)
        else:
            logger.info("Using existing model for training/retraining without rebuilding.")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Add custom history logger callback if filepath is provided
        if training_history_filepath:
            history_logger = TrainingHistoryLogger(training_history_filepath)
            callbacks.append(history_logger)
            logger.info(f"Training history will be logged to {training_history_filepath}")


        logger.info(f"Starting model training for {epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        logger.info("Training completed!")

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        if self.label_encoder is None:
             raise RuntimeError("Label encoder is not available for generating classification report.")

        y_pred_probs = self.model.predict(X_test)
        y_pred_classes = (y_pred_probs > 0.5).astype(int)

        target_names = self.label_encoder.classes_.tolist() if self.label_encoder is not None else []
        if not target_names:
            logger.warning("No target names available for classification report. Report might be incomplete.")
            report_dict = {}
        else:
            unique_classes_in_y_test = np.where(y_test.sum(axis=0) > 0)[0]
            if len(unique_classes_in_y_test) == 1:
                logger.warning("Only one unique class present in y_test for classification report. Adjusting target_names.")
                filtered_target_names = [target_names[i] for i in unique_classes_in_y_test]
                report_dict = classification_report(y_test, y_pred_classes, target_names=filtered_target_names, output_dict=True, zero_division=0)
            else:
                report_dict = classification_report(y_test, y_pred_classes, target_names=target_names, output_dict=True, zero_division=0)
        
        h_loss = hamming_loss(y_test, y_pred_classes)
        j_score = jaccard_score(y_test, y_pred_classes, average='samples')

        training_info = {
            "timestamp": datetime.now().isoformat(),
            "epochs_trained": len(self.history.history['loss']),
            "final_val_loss": float(self.history.history['val_loss'][-1]),
            "final_val_accuracy": float(self.history.history['val_accuracy'][-1]),
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
            "classification_report": report_dict,
            "hamming_loss": float(h_loss),
            "jaccard_score": float(j_score)
        }

        with open(os.path.join(self.model_dir, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        self.save_model()
        
        logger.info("Model and training info saved.")
        return training_info

    def save_model(self):
        """Save the trained Keras model and label encoder."""
        if self.model:
            model_path = os.path.join(self.model_dir, 'resistance_predictor_model.keras')
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        if self.label_encoder:
            encoder_path = os.path.join(self.data_dir, 'label_encoder.pkl')
            joblib.dump(self.label_encoder, encoder_path)
            logger.info(f"Label encoder saved to {encoder_path}")

    def load_model(self):
        """Load the trained Keras model and label encoder."""
        model_path = os.path.join(self.model_dir, 'resistance_predictor_model.keras')
        encoder_path = os.path.join(self.data_dir, 'label_encoder.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder file not found at {encoder_path}")

        self.model = keras.models.load_model(model_path)
        self.label_encoder = joblib.load(encoder_path)
        logger.info("Model and label encoder loaded.")

    def predict_from_fasta(self, fasta_file_path: str) -> Dict[str, Dict[str, float]]:
        """
        Predicts antibiotic resistance from a FASTA file.
        Returns a dictionary mapping sequence IDs to their resistance probabilities.
        """
        if self.model is None or self.label_encoder is None:
            raise RuntimeError("Model or label encoder not loaded. Train or load the model first.")
        
        if self.data_loader is None:
            raise RuntimeError("DataLoader instance not provided to ResistancePredictor. Cannot preprocess sequences.")

        sequences = []
        sequence_ids = []
        for record in SeqIO.parse(fasta_file_path, "fasta"):
            sequences.append(str(record.seq))
            sequence_ids.append(record.id)
        
        if not sequences:
            raise ValueError("No sequences found in the provided FASTA file.")

        features = np.array([self.data_loader.sequence_to_features(seq) for seq in sequences])
        
        prediction_probs = self.model.predict(features)

        results = {}
        for i, seq_id in enumerate(sequence_ids):
            probs_dict = {
                self.label_encoder.classes_[j]: float(prediction_probs[i][j])
                for j in range(len(self.label_encoder.classes_))
            }
            results[seq_id] = probs_dict
        
        return results

    def retrain(self, X_new: np.ndarray, y_new: np.ndarray, epochs: int = 10, mlb: Optional[MultiLabelBinarizer] = None):
        """Retrain the existing model with new data."""
        if self.model is None:
            raise RuntimeError("No existing model to retrain. Train a model first.")

        if mlb is not None:
            self.label_encoder = mlb
            logger.info("Using provided MultiLabelBinarizer instance for retraining.")
        elif self.label_encoder is None:
            try:
                self.label_encoder = joblib.load(os.path.join(self.data_dir, 'label_encoder.pkl'))
                logger.info("MultiLabelBinarizer loaded from disk for retraining.")
            except FileNotFoundError:
                logger.error("MultiLabelBinarizer not found. Retraining will proceed but dependent metrics might fail.")
        
        logger.info(f"Starting model retraining for {epochs} epochs with {len(X_new)} new samples.")

        history = self.model.fit(
            X_new, y_new,
            epochs=epochs,
            batch_size=16,
            callbacks=[
                EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001),
                ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, 'best_retrained_model.keras'),
                    monitor='loss',
                    save_best_only=True,
                    verbose=1
                )
            ],
            verbose=1
        )
        
        self.save_model()
        
        retrain_info = {
            'timestamp': datetime.now().isoformat(),
            'new_samples': int(len(X_new)),
            'epochs_retrained': len(history.history['loss']),
            'final_loss': float(history.history['loss'][-1])
        }
        
        with open(os.path.join(self.model_dir, 'retrain_info.json'), 'w') as f:
            json.dump(retrain_info, f, indent=2)
        
        logger.info("Retraining completed!")
        return retrain_info

    def evaluate_model(self, X_val: np.ndarray, y_val: np.ndarray, mlb: Optional[MultiLabelBinarizer] = None) -> Dict:
        """Evaluate the model and return metrics including classification report."""
        if self.model is None:
            raise RuntimeError("No model loaded for evaluation.")
        
        if mlb is not None:
            self.label_encoder = mlb
            logger.info("Using provided MultiLabelBinarizer instance for evaluation.")
        elif self.label_encoder is None:
            try:
                self.label_encoder = joblib.load(os.path.join(self.data_dir, 'label_encoder.pkl'))
                logger.info("MultiLabelBinarizer loaded from disk for evaluation.")
            except FileNotFoundError:
                logger.error("MultiLabelBinarizer not found. Cannot generate full classification report for evaluation.")
                return {"accuracy": "N/A", "f1_macro": "N/A", "hamming_loss": "N/A", "report": "Label encoder not found to generate report."}

        loss, accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        y_pred_probs = self.model.predict(X_val)
        y_pred_classes = (y_pred_probs > 0.5).astype(int)

        report_dict = {}
        if self.label_encoder and len(self.label_encoder.classes_) > 0:
            try:
                target_names = self.label_encoder.classes_.tolist()
                unique_classes_in_y_val = np.where(y_val.sum(axis=0) > 0)[0]
                if len(unique_classes_in_y_val) == 1:
                    logger.warning("Only one unique class present in y_val for classification report. Adjusting target_names.")
                    filtered_target_names = [target_names[i] for i in unique_classes_in_y_val]
                    report_dict = classification_report(y_val, y_pred_classes, target_names=filtered_target_names, output_dict=True, zero_division=0)
                else:
                    report_dict = classification_report(y_val, y_pred_classes, target_names=target_names, output_dict=True, zero_division=0)

            except Exception as e:
                logger.error(f"Error generating classification report: {e}")
                report_dict = {"error": f"Failed to generate report: {e}"}
        else:
            logger.warning("Cannot generate classification report: label encoder classes are missing or empty.")

        h_loss = hamming_loss(y_val, y_pred_classes)
        j_score = jaccard_score(y_val, y_pred_classes, average='samples')

        return {
            "accuracy": float(accuracy),
            "f1_macro": report_dict.get('macro avg', {}).get('f1-score', 'N/A'),
            "hamming_loss": float(h_loss),
            "jaccard_score": float(j_score),
            "report": json.dumps(report_dict, indent=2) if report_dict else "No report generated."
        }

if __name__ == "__main__":
    from data_loader import DataLoader
    import zipfile # Required for dummy zip creation

    loader = DataLoader()
    
    dummy_zip_path_folder_structure = "uploads/dummy_training_data_folder_structure.zip"
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    with zipfile.ZipFile(dummy_zip_path_folder_structure, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('ampicillin/seq1.fasta', '>seq1\nATGCGTACGT')
        zf.writestr('ampicillin/seq2.fasta', '>seq2\nTGCAACGTAC')
        zf.writestr('ciprofloxacin/seq3.fasta', '>seq3\nGCATCGTAGC')
        zf.writestr('ciprofloxacin/seq1.fasta', '>seq1\nATGCGTACGT')

    try:
        X_train, X_test, y_train, y_test, _, loaded_mlb = loader.load_data_from_uploaded_zip(
            dummy_zip_path_folder_structure, is_training_data=True, test_size=0.5
        )
        
        predictor = ResistancePredictor(data_loader_instance=loader)
        
        # Define a temporary file path for history logging
        temp_history_file = os.path.join(predictor.data_dir, "test_training_history.json")
        training_info = predictor.train(X_train, y_train, X_test, y_test, epochs=1, mlb=loaded_mlb, training_history_filepath=temp_history_file)
        
        # Test prediction
        dummy_predict_fasta = "uploads/dummy_predict.fasta"
        with open(dummy_predict_fasta, "w") as f:
            f.write(">test_seq1\nATGCGTACGT\n")
            f.write(">test_seq2\nGCATGCATGC\n")

        prediction_probs = predictor.predict_from_fasta(dummy_predict_fasta)
        
        print("Prediction probabilities for test sample:", prediction_probs)
        if prediction_probs:
            first_seq_id = list(prediction_probs.keys())[0]
            predicted_classes_for_first_sample = [
                drug for drug, prob in prediction_probs[first_seq_id].items() if prob >= 0.5
            ]
            print("Predicted resistance classes for test sample:", predicted_classes_for_first_sample)

    except Exception as e:
        print(f"Error during model test run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_zip_path_folder_structure):
            os.remove(dummy_zip_path_folder_structure)
        if os.path.exists(dummy_predict_fasta):
            os.remove(dummy_predict_fasta)
        # Clean up the temporary history file
        if os.path.exists(temp_history_file):
            os.remove(temp_history_file)

