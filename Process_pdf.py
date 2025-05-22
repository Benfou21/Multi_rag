import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from nomic import embed
import io

# Étape 1 : extraire les figures vectorielles
def extract_diagram_images(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    all_crops = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        drawings = page.get_drawings()  # formes vectorielles (lignes, courbes, path)
        bboxes = []

        for d in drawings:
            if d['type'] in ['path', 'curve', 'line']:  # heuristique simple
                bbox = fitz.Rect(d['rect'])
                bboxes.append(bbox)

        # Fusionner les bboxes proches (optionnel pour éviter les fragments)
        merged_bboxes = merge_bounding_boxes(bboxes)

        # Rendu image complet de la page
        pix = page.get_pixmap(dpi=dpi)
        page_img = Image.open(io.BytesIO(pix.tobytes("png")))

        for bbox in merged_bboxes:
            # Convertir bbox PDF en coordonnées pixels
            scale = dpi / 72
            rect_pix = [int(coord * scale) for coord in (bbox.x0, bbox.y0, bbox.x1, bbox.y1)]
            crop = page_img.crop(rect_pix)
            all_crops.append(crop)

    return all_crops

# Fusionne les boîtes qui se chevauchent (simple heuristique)
def merge_bounding_boxes(bboxes, threshold=20):
    if not bboxes:
        return []
    merged = []
    bboxes = sorted(bboxes, key=lambda r: r.y0)
    current = bboxes[0]

    for rect in bboxes[1:]:
        if current.intersects(rect) or current.distance_to(rect) < threshold:
            current |= rect  # union
        else:
            merged.append(current)
            current = rect
    merged.append(current)
    return merged

# Étape 2 : encoder avec Nomic
def encode_with_nomic(crops):
    model = embed.vision.get_model()
    tensors = [embed.vision.transforms.default_transform(img).unsqueeze(0) for img in crops]
    batch = torch.cat(tensors)
    return model.encode(batch)

# Pipeline
def process_pdf_with_pymupdf(pdf_path):
    crops = extract_diagram_images(pdf_path)
    print(f"{len(crops)} régions extraites.")
    if not crops:
        return None
    return encode_with_nomic(crops)

# Exemple d'utilisation
if __name__ == "__main__":
    pdf_path = "ton_fichier.pdf"
    embeddings = process_pdf_with_pymupdf(pdf_path)
    if embeddings is not None:
        print("Embedding shape:", embeddings.shape)
  
