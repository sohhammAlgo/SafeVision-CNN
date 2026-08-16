import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME","SafeVision")

MODEL_DIR = os.getenv("MODEL_DIR", "models")
DATA_DIR = os.getenv("DATA_DIR", "data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
LOG_DIR = os.getenv("LOG_DIR", "logs")

CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", 0.5)
)

IOU_THRESHOLD = float(
    os.getenv("IOU_THRESHOLD", 0.5)
)

WEBCAM_INDEX = int(
    os.getenv("WEBCAM_INDEX", 0)
)

FRAME_SKIP = int(
    os.getenv("FRAME_SKIP", 1)
)