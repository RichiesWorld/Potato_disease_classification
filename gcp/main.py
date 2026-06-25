import os
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

# Global variables for lazy loading
model = None

BUCKET_NAME = "richa-model-1"
class_names = ["Early Blight", "Late Blight", "Healthy"]


def download_saved_model_dir(bucket_name, source_dir, local_dir):
    """Downloads a folder structure from GCS to a local directory."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    # List all blobs under the specified prefix (folder)
    blobs = bucket.list_blobs(prefix=source_dir)

    for blob in blobs:
        # Create local path mirroring the GCS structure
        local_file_path = os.path.join(local_dir, os.path.relpath(blob.name, source_dir))

        # Ensure local directories exist
        local_file_dir = os.path.dirname(local_file_path)
        if not os.path.exists(local_file_dir):
            os.makedirs(local_file_dir)

        # Download file if it's not a placeholder directory blob
        if not blob.name.endswith('/'):
            blob.download_to_filename(local_file_path)


def predict(request):
    global model

    # Define local path where the model folder will be stored
    local_model_path = "/tmp/models/1"

    if model is None:
        print("Downloading model from GCS...")
        # Downloads everything inside 'models/1/' in GCS into '/tmp/models/1'
        download_saved_model_dir(BUCKET_NAME, "models/1/", local_model_path)

        print("Loading SavedModel into memory...")
        model = tf.keras.models.load_model(local_model_path)

    # Grab image from request
    image_file = request.files["file"]

    # Preprocess the image
    image = np.array(
        Image.open(image_file).convert("RGB").resize((256, 256))
    )
    image = image / 255.0  # Normalize

    # Fix: Use 'image' instead of the undefined 'img' variable
    img_array = tf.expand_dims(image, 0)

    # Run prediction
    predictions = model.predict(img_array)
    print("Predictions:", predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * float(np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}