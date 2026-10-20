
from pathlib import Path

from ultralytics import YOLO


# Update this path later to match your dataset YAML in Google Drive.
DATASET_YAML = "/content/drive/MyDrive/MarineWaste-18K/data.yaml"

MODEL_NAME = "yolov8n.pt"
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16


def main():
    if not Path(DATASET_YAML).is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATASET_YAML}\n"
            "Mount Google Drive in Colab and update DATASET_YAML "
            "to the actual location of your data.yaml file."
        )

    model = YOLO(MODEL_NAME)

    # Standard YOLOv8 training baseline.
    # AMCA integration will be added and tested separately.
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=0,
        workers=2,
        project="runs/underwater_waste",
        name="yolov8_baseline",
        seed=42,
    )


if __name__ == "__main__":
    main()
