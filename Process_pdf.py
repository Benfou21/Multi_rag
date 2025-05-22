import cv2
import torch
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from nomic import embed

# === Étape 1 : Render PDF to images ===
def render_pdf_to_images(pdf_path, dpi=200):
    return convert_from_path(pdf_path, dpi=dpi)

# === Étape 2 : Load YOLO model for layout detection ===
def load_doclayout_model(model_path="weights/yolov8_doclaynet.pt"):
    return YOLO(model_path)

# === Étape 3 : Detect figures ===
def detect_figures_yolo(image, model, conf=0.5):
    results = model.predict(source=np.array(image), conf=conf, imgsz=1280, verbose=False)[0]
    boxes = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        label = model.names[int(cls)]
        if label.lower() == "figure":  # on peut aussi extraire "table", "equation", etc.
            boxes.append([int(coord) for coord in box])
    return boxes

# === Étape 4 : Crop images ===
def crop_boxes(image, boxes):
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crops.append(image.crop((x1, y1, x2, y2)))
    return crops

# === Étape 5 : Encode with Nomic ===
def encode_with_nomic_vision(crops):
    model = embed.vision.get_model()
    tensors = [embed.vision.transforms.default_transform(img).unsqueeze(0) for img in crops]
    return model.encode(torch.cat(tensors))

# === Pipeline principal ===
def process_pdf_with_yolo(pdf_path):
    page_images = render_pdf_to_images(pdf_path)
    yolo_model = load_doclayout_model()

    all_crops = []

    for img in page_images:
        boxes = detect_figures_yolo(img, yolo_model)
        crops = crop_boxes(img, boxes)
        all_crops.extend(crops)

    print(f"{len(all_crops)} figures détectées.")

    if not all_crops:
        return None

    return encode_with_nomic_vision(all_crops)

# === Exemple d'utilisation ===
if __name__ == "__main__":
    embeddings = process_pdf_with_yolo("ton_fichier.pdf")
    if embeddings is not None:
        print("Shape des embeddings :", embeddings.shape)
