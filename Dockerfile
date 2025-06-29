# Use a Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create necessary directories at startup if they don't exist
# This helps ensure the app has places to store models and data, even in a fresh container
RUN mkdir -p /app/uploads /app/data /app/models

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application using uvicorn
# The --host 0.0.0.0 makes the server accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]