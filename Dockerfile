# Use a lightweight Python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the necessary files
COPY requirements.txt requirements.txt
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app will run on
EXPOSE 5000

# Run the Flask app
CMD ["python", "src/api.py"]
