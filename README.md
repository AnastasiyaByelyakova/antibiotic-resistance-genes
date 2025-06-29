# Bacterial Drug Resistance Predictor

## Project Overview

The **Bacterial Drug Resistance Predictor** is a machine learning pipeline built with FastAPI and TensorFlow/Keras designed to predict antibiotic resistance from bacterial genomic sequences (FASTA files). This application features **user authentication**, requiring users to log in before accessing the prediction and monitoring functionalities. It provides a web-based interface for users to upload genomic data, train/retrain models, validate model performance, and view real-time prediction results and model statistics.

## Features

* **User Authentication**: Secure login and registration system. All core functionalities (prediction, dashboard, training, validation) now require a logged-in user.

* **Antibiotic Resistance Prediction:** Upload FASTA files containing bacterial genomic sequences to get predictions on resistance to various antibiotic classes.

* **Model Training:** Train a new deep learning model from scratch using a provided dataset (ZIP archive containing FASTA files and a `metadata.json` for labels).

* **Model Retraining:** Update an existing model with new data to improve its performance or adapt to new resistance patterns.

* **Model Validation:** Evaluate the loaded model's performance on a separate validation dataset, providing metrics like accuracy, F1-score (macro), and Hamming loss.

* **Interactive Dashboard:** A comprehensive dashboard to monitor model status, training statistics, recent predictions, frequent resistances, and real-time training progress (via chart).

* **Model Summary:** View the detailed architecture summary of the loaded Keras model.

* **Data Handling:** Utilizes `BioPython` for sequence parsing and `scikit-learn` for multi-label binarization of antibiotic classes.

## Setup and Installation

### Prerequisites

* Python 3.8+

* MySQL Database (must be running)

* Git (optional, for cloning the repository)

### 1. Clone the Repository (if applicable)

If you have a Git repository, clone it and navigate into the project directory:

```bash
git clone <repository_url>
cd bacterial-resistance-predictor
```

### 2. Set up a Python Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies:

```bash
git clone <repository_url>
cd bacterial-resistance-predictor
```


Markdown

# Bacterial Drug Resistance Predictor

## Project Overview

The **Bacterial Drug Resistance Predictor** is a machine learning pipeline built with FastAPI and TensorFlow/Keras designed to predict antibiotic resistance from bacterial genomic sequences (FASTA files). This application features **user authentication**, requiring users to log in before accessing the prediction and monitoring functionalities. It provides a web-based interface for users to upload genomic data, train/retrain models, validate model performance, and view real-time prediction results and model statistics.

## Features

* **User Authentication**: Secure login and registration system. All core functionalities (prediction, dashboard, training, validation) now require a logged-in user.

* **Antibiotic Resistance Prediction:** Upload FASTA files containing bacterial genomic sequences to get predictions on resistance to various antibiotic classes.

* **Model Training:** Train a new deep learning model from scratch using a provided dataset (ZIP archive containing FASTA files and a `metadata.json` for labels).

* **Model Retraining:** Update an existing model with new data to improve its performance or adapt to new resistance patterns.

* **Model Validation:** Evaluate the loaded model's performance on a separate validation dataset, providing metrics like accuracy, F1-score (macro), and Hamming loss.

* **Interactive Dashboard:** A comprehensive dashboard to monitor model status, training statistics, recent predictions, frequent resistances, and real-time training progress (via chart).

* **Model Summary:** View the detailed architecture summary of the loaded Keras model.

* **Data Handling:** Utilizes `BioPython` for sequence parsing and `scikit-learn` for multi-label binarization of antibiotic classes.

## Setup and Installation

### Prerequisites

* Python 3.8+

* MySQL Database (must be running)

* Git (optional, for cloning the repository)

### 1. Clone the Repository (if applicable)

If you have a Git repository, clone it and navigate into the project directory:

```bash
git clone <repository_url>
cd bacterial-resistance-predictor
```

### 2. Set up a Python Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies:


```bash
python -m venv venv
source venv/bin/activate   # On Windows: `venv\Scripts\activate`
```

### 3. Install Dependencies
You need to ensure all required Python libraries are installed.

```bash
pip install -r requirements.txt
```


### 4. MySQL Database Setup
Ensure your MySQL server is running.

#### a. Create the database and tables 
Run initiate.sql script to create a database and tables. 

#### c. Add Initial Users (Crucial for Login)

Use this python code to get a hash code for your passwords.

```python
from passlib.hash import bcrypt

# Example for a new user 'your_username' with password 'your_secure_password'
plain_password = "your_secure_password"
hashed_password = bcrypt.hash(plain_password, rounds=12) # Use a suitable number of rounds
print(f"Hashed password for '{plain_password}': {hashed_password}")
```

And use this hash value to users table in database. 

```sql

INSERT INTO users (username, hashed_password) VALUES
('your_username', 'PASTE_THE_GENERATED_HASH_HERE');
```


Important: Replace PASTE_THE_GENERATED_HASH_HERE with the actual hash you obtained from python in previous step.

### 5. Configure Environment Variables (.env)
Create a file named .env in the root directory of your project. This file will store your database connection details and a crucial secret key for handling user sessions (JWT tokens).

```bash 
MYSQL_HOST=localhost
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=bdr_predictor_db
SECRET_KEY="a_very_long_and_random_secret_key_for_jwt_tokens_generate_this_with_openssl_rand_hex_32"
```

Replace your_mysql_user, your_mysql_password, and bdr_predictor_db with your actual MySQL credentials and database name.

Generate a strong SECRET_KEY: Use a command like openssl rand -hex 32 in your terminal and copy the output string here. This key is vital for JWT security.

### 6. Run the Application
Once all setup steps are complete, you can start the FastAPI application from your project's root directory:

```bash

uvicorn main:app --reload
```

The --reload flag is helpful during development as it automatically restarts the server when code changes are detected.

The application will now be accessible at http://localhost:8000.

## Data Format for Model Training

The model expects training data to be provided as a ZIP archive (.zip file). 

### Folder-based Labeling

In this approach, each folder within the ZIP file represents an antibiotic resistance class, and all FASTA files directly inside that folder are assumed to belong to that class. A single FASTA file can be present in multiple folders if the sequence exhibits resistance to multiple antibiotics.

ZIP File Structure:

your_training_data.zip
├── ampicillin/
│   ├── sequence1.fasta
│   ├── sequence2.fasta
│   └── ...
├── ciprofloxacin/
│   ├── sequence3.fasta
│   ├── sequence1.fasta  # Same sequence can be in multiple folders
│   └── ...
├── tetracycline/
│   ├── sequence4.fasta
│   └── ...
└── ...

ampicillin/: A folder named after an antibiotic class (e.g., 'ampicillin').

sequence1.fasta, sequence2.fasta: FASTA files containing bacterial genomic sequences. Each file can contain one or more sequences. The DataLoader will parse all sequences within these files.

Multi-labeling: If a sequence (e.g., sequence1.fasta) is found in both the ampicillin/ and ciprofloxacin/ folders, it will be labeled as resistant to both 'ampicillin' and 'ciprofloxacin'.



## Usage

### Access the Application: 

Open your web browser and navigate to http://localhost:8000.

### Login / Register: 

You will be automatically redirected to the login page (/login).

If you don't have an account, click "Register here" to create a new one. 
After successful registration, you will be taken back to the login page.

### Predict Resistance:

Once logged in, you will be on the "Predict" page.

Select a FASTA file (.fasta, .fa, or .fna) containing bacterial genomic sequences.

Click "Predict Resistance".

The prediction results will be displayed on the page, showing the predicted resistances for each sequence. The name of the predicted file will be displayed.

You can also click "Download Results (JSON)" to save the full prediction output.

### View Dashboard:

Navigate to the "Dashboard" using the link in the top navigation bar.

The dashboard provides comprehensive statistics on model performance, recent prediction activities, frequent resistances, and real-time training progress (via a chart).

Key metrics like Training Samples, Test Samples, and Model Accuracy will update periodically without requiring a full page reload.

#### Train/Retrain Model:

From the "Dashboard", you can access sections to upload a ZIP file containing training data (structured by antibiotic class folders or with metadata.json).

Use this to train a new deep learning model from scratch or retrain an existing one with new data.

### Validate Model:

Also from the "Dashboard", you can upload a validation ZIP file to evaluate the currently loaded model's performance on a separate dataset.

### Logout: 

To end your session, click on your username in the top right corner of the navbar and select "Logout". You will be redirected to the login page.

