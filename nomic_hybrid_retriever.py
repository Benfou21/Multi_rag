from __future__ import annotations
import math
from typing import List, Optional, Union, cast
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
import pytesseract
from tqdm import tqdm
import imagehash
from benchs.utils.iter import batched
from benchs.utils.torch_utils import get_torch_device

from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

class NomicHybridRetriever():
    """
    Retriever hybride qui combine :
      - embeddings texte extraits par OCR + nomic-embed-text
      - embeddings visuels extraits des diagrammes/images par nomic-embed-vision
    Les embeddings sont fusionnés via une pondération alpha, et
    on calcule ensuite la similarité par produit scalaire.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        pure : bool = False,
        device: str = "auto",
        **kwargs,
    ):
        self.pure = pure
        self.alpha = alpha
        self.device = get_torch_device(device)

        # Modèles Nomic Embed Text & Vision
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5"
        )
        self.text_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True
        ).to(self.device)

        self.image_processor = AutoImageProcessor.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5"
        )
        self.vision_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5",
            trust_remote_code=True
        ).to(self.device)

        # Modèle de détection de layout pour extraire diagrammes/images
        filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
        self.layout_model = YOLOv10(filepath)

    @staticmethod
    def _mean_pooling(model_output: Tensor, attention_mask: Tensor) -> Tensor:
        token_embeddings = model_output[0]  # last_hidden_state
        mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        return (
            torch.sum(token_embeddings * mask_expanded, dim=1)
            / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        )


    def forward_queries(
        self,
        queries: List[str],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        self.text_model.eval()
        list_emb: List[torch.Tensor] = []

        for batch in tqdm(batched(queries, batch_size),
                          desc="Embedding queries", leave=False):
            texts = ["search_query: " + q for q in cast(List[str], batch)]
            enc = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.text_model(**enc)
                pooled = self._mean_pooling(out, enc["attention_mask"])
                pooled = F.layer_norm(pooled, (pooled.shape[1],))
                pooled = F.normalize(pooled, p=2, dim=1)

            list_emb.extend(torch.unbind(pooled, dim=0))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return list_emb

    def forward_passages(
        self,
        passages: List[Image.Image],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        
        # Not Pure img text
        # 1️⃣ OCR + embedding texte
        if not self.pure : 
            text_embs: List[torch.Tensor] = []
            self.text_model.eval()
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
                    text_embs.extend([torch.zeros(self.text_model.config.hidden_size) for _ in range(num_empty)])
                    continue

                enc = self.tokenizer(
                    list(text_batch), padding=True, truncation=True, return_tensors="pt", max_length=8192 # Nomic max length
                ).to(self.device)
                with torch.no_grad():
                    out = self.text_model(**enc)
                    pooled = self._mean_pooling(out, enc["attention_mask"])
                    pooled = F.layer_norm(pooled, normalized_shape=(pooled.shape[1],), eps=1e-7)
                    pooled = F.normalize(pooled, p=2, dim=1)
                text_embs.extend(torch.unbind(pooled.cpu(), dim=0))
                del enc, out, pooled
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()



        # Extraction des diagrammes/images (si pas pure) de tout (si pure) 
        total_crops: List[List[Image.Image]] = []
        self.vision_model.eval() 

        for page in tqdm(passages, desc="Layout extracation page images", leave=False):
            det = self.layout_model.predict(
                page, imgsz=1024, conf=0.2, device=self.device
            )
            crops: List[Image.Image] = []
            if det and len(det[0].boxes) > 0:
                for box in det[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(page.width, x2)
                    y2 = min(page.height, y2)
                    class_id = int(box.cls[0])
                    class_name = self.layout_model.names[class_id]

                    #En pure mode on prend chaque layout en image sinon que les diagrame
                    if self.pure or class_name in ["figure","table"]:
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
            proc = self.image_processor(
                images=rgb_page_crops, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.vision_model(**proc)
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

        if not self.pure:
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

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor] ], # Expected shape: (num_queries, D) or (D,) if single query, or List[torch.Tensor]
        passage_embeddings: List[List[torch.Tensor]], # Expected from forward_passages: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Calculates scores for each page against each query.
        For each query and each page, the score is the maximum cosine similarity
        between that query embedding and any individual embedding (text or image crop)
        associated with that page.

        Args:
            query_embeddings (torch.Tensor or List[torch.Tensor]): The embeddings of queries.
                                             If torch.Tensor, expected shape (num_queries, D) or (D,).
                                             If List[torch.Tensor], it will be stacked into (num_queries, D).
            passage_embeddings (List[List[torch.Tensor]]): A list where each inner list
                                                            contains all embeddings (text + image crops)
                                                            for a single page.

        Returns:
            torch.Tensor: A tensor containing scores for each query against each page.
                          Shape (num_queries, num_pages).
        """
        # Ensure query_embeddings is a torch.Tensor and has shape (num_queries, D)
        if isinstance(query_embeddings, list):
            if not query_embeddings: # Handle empty list of queries
                print("Warning: query_embeddings is an empty list. Returning empty scores.")
                return torch.tensor([])
            # Stack list of tensors into a single tensor
            query_embeddings = torch.stack(query_embeddings)
        
        # Now query_embeddings is guaranteed to be a torch.Tensor
        # If a single query is passed as (D,), unsqueeze to (1, D)
        if query_embeddings.dim() == 1:
            query_embeddings = query_embeddings.unsqueeze(0)

        num_queries = query_embeddings.shape[0]
        per_query_page_scores: List[torch.Tensor] = [] # This will store (num_queries,) tensors

        for i, page_embedding_list in enumerate(passage_embeddings):
            if not page_embedding_list:
                # If a page has no embeddings, assign a very low score for all queries.
                # This ensures it won't be ranked highly unless all other pages also have no embeddings.
                per_query_page_scores.append(torch.full((num_queries,), -float('inf')).to(query_embeddings.device))
                continue

            # Stack all embeddings for the current page into a single tensor.
            # page_tensor will have shape (num_elements_in_page, D).
            # Move to the same device as query_embeddings for computation.
            page_tensor = torch.stack(page_embedding_list).to(query_embeddings.device)

            # Calculate cosine similarity between all query embeddings and all page element embeddings.
            # query_embeddings.unsqueeze(1) -> (num_queries, 1, D)
            # page_tensor.unsqueeze(0) -> (1, num_elements_in_page, D)
            # F.cosine_similarity(..., dim=2) computes similarity along the last dimension (D).
            # The result 'similarities' will have shape (num_queries, num_elements_in_page).
            similarities = F.cosine_similarity(query_embeddings.unsqueeze(1), page_tensor.unsqueeze(0), dim=2)

            # For each query, get the maximum similarity score against any element in the current page.
            # max_similarities_for_page will have shape (num_queries,).
            max_similarities_for_page, _ = torch.max(similarities, dim=1)
            per_query_page_scores.append(max_similarities_for_page)

        # Stack the list of (num_queries,) tensors into a (num_queries, num_pages) tensor.
        if not per_query_page_scores:
            return torch.tensor([]) # Return an empty tensor if no pages were processed.
        scores = torch.stack(per_query_page_scores, dim=1) # Stack along a new dimension for pages

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return scores
