import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError(
        "ROBOFLOW_API_KEY not found. "
        "Add it to your .env file."
    )

WORKSPACE = "ppe-detection-f9rby"
PROJECT = "ppe-detection-yolo-11-5paei"
VERSION = 1

DOWNLOAD_FORMAT = "yolov8"

print("Connecting to Roboflow...")

rf = Roboflow(api_key=API_KEY)

project = rf.workspace(WORKSPACE).project(PROJECT)

version = project.version(VERSION)

print("Downloading PPE dataset...")

dataset = version.download(DOWNLOAD_FORMAT)

print("\nDataset downloaded successfully!")
print(f"Dataset location: {dataset.location}")

#curl -L "https://universe.roboflow.com/ds/YSKHtrBlJp?key=uN2aUaaKb4" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
#https://universe.roboflow.com/ds/YSKHtrBlJp?key=uN2aUaaKb4