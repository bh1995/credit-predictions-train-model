# Use an official Python runtime as a base image
FROM python:3.10-slim
# Set the working directory in the container to /app
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app
# Install the dependencies
RUN pip install -r requirements.txt
# Define environment variable
ENV NAME World
# Run main.py when the container launches
CMD ["python", "main.py"]

