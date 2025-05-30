import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from datasets import load_dataset
from PIL import Image
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
import pytesseract
import matplotlib.pyplot as plt

# --- Configuration ---
DATASET = "nielsr/vitore_benchmark"
SPLIT = "test"
BATCH_SIZE = 8
TOP_K = 5  # number of top results to inspect
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Models ---
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
text_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(DEVICE)

image_processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True).to(DEVICE)

# YOLO detector for layout (figures and tables)
model_path = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                             filename="doclayout_yolo_docstructbench_imgsz1024.pt")
yolo = YOLOv10(model_path)

# --- Helper Functions ---
def mean_pooling(outputs, mask):
    tokens = outputs.last_hidden_state
    mask = mask.unsqueeze(-1).expand(tokens.size()).float()
    return torch.sum(tokens * mask, dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def embed_texts(texts):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = text_model(**enc)
        pooled = mean_pooling(out, enc.attention_mask)
        emb = F.normalize(F.layer_norm(pooled, pooled.shape[-1:]), p=2)
    return emb.cpu()


def extract_crops(image):
    det = yolo.predict(image, imgsz=1024, conf=0.2, device=DEVICE)
    crops = []
    if det and det[0].boxes:
        for box in det[0].boxes:
            cls = yolo.names[int(box.cls[0])]
            if cls in ['figure', 'table']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crops.append(image.crop((x1, y1, x2, y2)))
    return crops or [image]


def forward_passages(
        passages: List[Image.Image],
        batch_size: int,
        pure: bool,
        **kwargs,
    ) -> List[torch.Tensor]:
        
        # Not Pure img text
        # 1️⃣ OCR + embedding texte
        if not pure : 
            text_embs: List[torch.Tensor] = []
            text_model.eval()
            ocr_texts: List[str] = []
            for page_img in tqdm(passages, desc="Performing OCR", leave=False, disable=len(passages)<5):
                 try:
                    ocr_texts.append(pytesseract.image_to_string(page_img))
                 except Exception as e:
                    print(f"Warning: Pytesseract OCR failed for a page: {e}. Using empty string.")
                    ocr_texts.append("")

            for text_batch in tqdm(
                batched(ocr_texts, batch_size), 
                desc="Embedding page texts", 
                leave=False,
                total= (len(ocr_texts) + batch_size -1) // batch_size,
                disable=len(ocr_texts)<batch_size*2
            ):
                if not any(text_batch): 
                    num_empty = len(list(text_batch))
                    text_embs.extend([torch.zeros(text_model.config.hidden_size) for _ in range(num_empty)])
                    continue

                enc = tokenizer(
                    list(text_batch), padding=True, truncation=True, return_tensors="pt", max_length=8192 # Nomic max length
                ).to(device)
                with torch.no_grad():
                    out = text_model(**enc)
                    pooled = _mean_pooling(out, enc["attention_mask"])
                    pooled = F.layer_norm(pooled, normalized_shape=(pooled.shape[1],), eps=1e-7)
                    pooled = F.normalize(pooled, p=2, dim=1)
                text_embs.extend(torch.unbind(pooled.cpu(), dim=0))
                del enc, out, pooled
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()



        # Extraction des diagrammes/images (si pas pure) de tout (si pure) 
        total_crops: List[List[Image.Image]] = []
        vision_model.eval() 

        for page in tqdm(passages, desc="Layout extracation page images", leave=False):
            det = layout_model.predict(
                page, imgsz=1024, conf=0.2, device=device
            )
            crops: List[Image.Image] = []
            if det and len(det[0].boxes) > 0:
                for box in det[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(page.width, x2)
                    y2 = min(page.height, y2)
                    class_id = int(box.cls[0])
                    class_name = layout_model.names[class_id]

                    #En pure mode on prend chaque layout en image sinon que les diagrame
                    if pure or class_name in ["figure","table"]:
                        crops.append(page.crop((x1, y1, x2, y2)))
                        plt.imshow(page.crop((x1, y1, x2, y2)))
                        plt.show()

            # Si pas de diagramme détecté, fallback sur la page entière
            if len(crops) == 0:
                crops = [page]
            total_crops.append(crops)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        

        #Image embeddings    
        img_embs: List[List[torch.Tensor]] = [] # Stores list of embeddings for each page
        for page_crops in tqdm(total_crops, desc="Embedding per page images", leave=False):
            current_page_img_embs: List[torch.Tensor] = []
            if not page_crops:
                img_embs.append([])
                continue

            # Convert images to RGB if necessary for the processor
            rgb_page_crops = [crop.convert("RGB") if crop.mode != "RGB" else crop for crop in page_crops]

            # Process image crops and get embeddings
            proc = image_processor(
                images=rgb_page_crops, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                out = vision_model(**proc)
                # Mean pool over patches to get a single embedding per crop
                feats = out.last_hidden_state.mean(dim=1)
                feats = F.layer_norm(feats, (feats.shape[1],))
                feats = F.normalize(feats, p=2, dim=1)

            current_page_img_embs.extend(torch.unbind(feats.cpu(), dim=0)) # Unbind to get individual tensors
            img_embs.append(current_page_img_embs) # Append the list of embeddings for the current page

            del page_crops, rgb_page_crops, proc, out, feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        
        final_page_embeddings: List[List[torch.Tensor]] = []
        num_pages = len(passages)

        if not pure:
            if len(text_embs) != num_pages or len(img_embs) != num_pages:
                print(f"Warning: Mismatch in number of text ({len(text_embs)}) and image ({len(img_embs)}) embedding lists. Expected {num_pages}.")

            for i in range(num_pages):
                current_page_combined_embs = []
                # Add text embedding for the current page if available
                if i < len(text_embs) and text_embs[i] is not None:
                    current_page_combined_embs.append(text_embs[i])
                # Add all image embeddings for the current page if available
                if i < len(img_embs):
                    current_page_combined_embs.extend(img_embs[i])
                final_page_embeddings.append(current_page_combined_embs)
        else:
            # If in pure image mode, only return the image embeddings
            final_page_embeddings = img_embs

        return final_page_embeddings


def retrieve_topk(query_emb, corpus_embs, k):
    sims = [F.cosine_similarity(query_emb.unsqueeze(0), doc_embs, dim=1).max().item()
            for doc_embs in corpus_embs]
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def analyze_failures(queries, qrels,ds_c, corpus_embs, k=TOP_K):
    for qi, q in enumerate(queries):
        print(f"\nQuery {qi}: {q}")
        top = retrieve_topk(query_embs[qi], corpus_embs, k)
        gold = set(qrels.get(qi, []))
        print(f"  Relevant docs: {gold}")
        for rank, (doc_id, score) in enumerate(top, 1):
            flag = '✔' if doc_id in gold else '✖'
            print(f"  {rank}. Doc {doc_id} (score: {score:.3f}) {flag}")
            img = ds_c[doc_id]["image"]
            plt.figure(figsize=(4,4))
            plt.imshow(img); plt.axis('off')
        input("Press Enter for next query...")

# --- Main Script ---
if __name__ == '__main__':
    # 1. Load dataset
    ds_c = load_dataset(DATASET, name='corpus', split=SPLIT)
    ds_q = load_dataset(DATASET, name='queries', split=SPLIT)
    ds_r = load_dataset(DATASET, name='qrels', split=SPLIT)

    queries = [ex['query'] for ex in ds_q]
    qrels = {ex['query-id']: [] for ex in ds_r}
    for ex in ds_r:
        qrels.setdefault(ex['query-id'], []).append(ex['corpus-id'])


    
    # 2. Embed corpus
    print("Embedding corpus...")
    all_passages_embs: List[Tensor] = []
    num_corpus_docs = len(ds_c)
  
    print(f"\nEncoding pages...")
    for i in tqdm(range(0, num_corpus_docs, corpus_processing_batch_size), desc="Corpus Encoding Batches"):
        batch_start = i
        batch_end = min(i + corpus_processing_batch_size, num_corpus_docs)
            
        current_batch_dataset_slice = ds_c.select(range(batch_start, batch_end))
        current_batch_images: List[Image.Image] = [ex['image'] for ex in current_batch_dataset_slice]
        all_passages_embs.append(forward_passages(current_batch_images))
    
    
    # 3. Embed queries
    print("Embedding queries...")
    query_embs = embed_texts(queries)

    # 4. Analyze failures
    analyze_failures(queries, qrels, ds_c, corpus_embs)
