from __future__ import annotations
import math
from typing import List, Optional, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
import pytesseract
from tqdm import tqdm

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device

from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from doclayout_yolo import YOLOv10


@register_vision_retriever("nomic_hybrid_retriever")
class NomicHybridRetriever(BaseVisionRetriever):
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
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)
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
        self.layout_model = YOLOv10()

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

        return list_emb

    def forward_passages(
        self,
        passages: List[Image.Image],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        # 1️⃣ OCR + embedding texte
        texts: List[str] = [
            pytesseract.image_to_string(page) for page in passages
        ]
        text_embs: List[torch.Tensor] = []
        for batch in tqdm(
            batched(texts, batch_size),
            desc="Embedding page texts",
            leave=False,
        ):
            enc = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.text_model(**enc)
                pooled = self._mean_pooling(out, enc["attention_mask"])
                pooled = F.layer_norm(pooled, (pooled.shape[1],))
                pooled = F.normalize(pooled, p=2, dim=1)
            text_embs.extend(torch.unbind(pooled, dim=0))

        # 2️⃣ Extraction des diagrammes/images + embedding vision
        img_embs: List[torch.Tensor] = []
        for page in tqdm(passages, desc="Embedding page images", leave=False):
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
                    crops.append(page.crop((x1, y1, x2, y2)))

            # Si pas de diagramme détecté, fallback sur la page entière
            if len(crops) == 0:
                crops = [page]

            # Embedding des crops
            proc = self.image_processor(
                images=crops, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.vision_model(**proc)
                # moyenne sur les patches
                feats = out.last_hidden_state.mean(dim=1)
                feats = F.layer_norm(feats, (feats.shape[1],))
                feats = F.normalize(feats, p=2, dim=1)
            # moyenne sur les crops
            img_embs.append(feats.mean(dim=0))

        # 3️⃣ Fusion texte + vision
        assert len(text_embs) == len(img_embs)
        combined: List[torch.Tensor] = [
            self.alpha * t + (1.0 - self.alpha) * v
            for t, v in zip(text_embs, img_embs)
        ]
        return combined

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        # Convertit en Tensor [n_queries, dim] et [n_docs, dim]
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings)

        # Score par produit scalaire
        # -> [n_queries, n_docs]
        scores = torch.matmul(
            query_embeddings, passage_embeddings.T
        )
        return scores
