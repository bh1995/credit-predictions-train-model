import datetime
import time
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import json
import io
from io import StringIO
import pickle

def download_csv_from_blob(container_name, blob_name):
    # Get the connection string from the environment variable
    connect_str = "DefaultEndpointsProtocol=https;AccountName=creditproject;AccountKey=ki155kFi7Q5RgnaOCui+rRKqKFW9D/8n9SL90GnCa9ZTNg8sVBdZC35wg0Y1CxC392oCLXkoBpRB+AStebLk7w==;EndpointSuffix=core.windows.net"
    # Create a BlobServiceClient using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a BlobClient to handle the CSV blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    # Download the blob content
    blob_data = blob_client.download_blob()
    data = blob_data.readall()
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(StringIO(data.decode('utf-8')))

    return df

def train_model(df):
    # Train XGBoost model and fit label_encoder
    # Encode categorical data
    label_encoder = LabelEncoder()
    df['purpose'] = label_encoder.fit_transform(df['purpose'])
    # Separate the features and the target
    X = df.drop('not.fully.paid', axis=1)
    y = df['not.fully.paid']
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    # Make predictions
    predictions = model.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")
    
    return model, label_encoder

def save_model_to_blob(model, label_encoder, container_name, blob_names):
    # Save the model to Blob Storage
    connect_str = "DefaultEndpointsProtocol=https;AccountName=creditproject;AccountKey=ki155kFi7Q5RgnaOCui+rRKqKFW9D/8n9SL90GnCa9ZTNg8sVBdZC35wg0Y1CxC392oCLXkoBpRB+AStebLk7w==;EndpointSuffix=core.windows.net"
    
    # Serialize the Model In-Memory
    # Assuming 'model' is your XGBoost model
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)  # Rewind the buffer to the beginning
    # Create a BlobServiceClient using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Get a reference to the container you want to upload to
    container_client = blob_service_client.get_container_client(container_name)
    # Create a blob client using the container_client
    blob_client = container_client.get_blob_client(blob_names[0])
    # Upload the stream
    blob_client.upload_blob(buffer, overwrite=True)
    print(f"Stream successfully uploaded to {blob_names[0]} in container {container_name}.")

    # Save the fitted LabelEncoder to a pickle file in blob storage
    # Upload the file locally
    blob_client = container_client.get_blob_client(blob_names[1])
    with open(blob_names[1], "wb") as file:
        pickle.dump(label_encoder, file)
    with open(blob_names[1], "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"File label_encoder.pkl successfully uploaded in container {container_name}.")


def rename_old_model(container_name, blob_name, extension):
    # Get the connection string from the environment variable
    connect_str = "DefaultEndpointsProtocol=https;AccountName=creditproject;AccountKey=ki155kFi7Q5RgnaOCui+rRKqKFW9D/8n9SL90GnCa9ZTNg8sVBdZC35wg0Y1CxC392oCLXkoBpRB+AStebLk7w==;EndpointSuffix=core.windows.net"
    # Create a BlobServiceClient using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)
    # Create a new blob name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    new_blob_name = f"{blob_name}_{timestamp}.{extension}"
    # Copy the existing blob to the new blob
    source_blob = f"https://creditproject.blob.core.windows.net/{container_name}/{blob_name}.{extension}"
    print("source_blob", source_blob)
    copied_blob = container_client.get_blob_client(new_blob_name)
    copied_blob.start_copy_from_url(source_blob)

    # Check if the copy is completed
    for _ in range(10):  # Retry 10 times
        props = copied_blob.get_blob_properties()
        if props.copy.status != 'pending':
            break
        time.sleep(1)
    else:
        raise Exception("Timeout: Failed to copy blob within the expected time.")

    # If copy is successful, delete the old blob
    if props.copy.status == 'success':
        old_blob = container_client.get_blob_client(blob_name)
        old_blob.delete_blob()

    return new_blob_name



if __name__ == "__main__":
    df = download_csv_from_blob(container_name='data', blob_name='loan_data.csv')
    print(df)
    model, label_encoder = train_model(df)
    # new_blob_name = rename_old_model(container_name='models', blob_name='model', extension='xgb')
    # print(f"Blob renamed to: {new_blob_name}")
    # new_blob_name = rename_old_model(container_name='models', blob_name='label_encoder', extension='pkl')
    # print(f"Blob renamed to: {new_blob_name}")
    save_model_to_blob(model=model, label_encoder=label_encoder, container_name='models', blob_names=('model.xgb', 'label_encoder.pkl'))



