# Base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the Flask application code to the container
COPY . .

# Expose the port on which the Flask application will run
EXPOSE 8888

# Start the Flask application
CMD ["python", "app.py"]
