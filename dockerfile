# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8000 to allow access to the FastAPI application
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]
