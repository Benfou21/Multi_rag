from __future__ import annotations
from typing import List, Optional, Union

import torch
from PIL import Image
import pytesseract
from nomic import embed

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever


@register_vision_retriever("nomic_hybrid_retriever")
class NomicHybridRetriever(BaseVisionRetriever):
    """
    Retriever hybride qui combine :
      - embeddings texte extraits par OCR + nomic.embed.text
      - embeddings visuels par nomic.embed.image
    Les scores sont fusionnés via une pondération alpha.
    """
    def __init__(
        self,
        alpha: float = 0.5,
        text_model: str = "nomic-embed-text-v1.5",
        image_model: str = "nomic-embed-vision-v1.5",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)
        self.alpha = alpha
        self.text_model = text_model
        self.image_model = image_model
        self.device = device

    def forward_queries(
        self,
        queries: List[str],
        batch_size: int,
        **kwargs,
    ) -> torch.Tensor:
        # Embedding textuel des queries
        # nomic.embed.text renvoie List[List[float]]
        text_embs = embed.text(
            texts=queries,
            model=self.text_model,
            task_type="search_query"
        )
        tensor = torch.tensor(text_embs, dtype=torch.float32, device=self.device)
        return tensor

    def forward_passages(
        self,
        passages: List[Image.Image],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        # 1) OCR pour extraire le texte de chaque page
        docs_text = [pytesseract.image_to_string(img) for img in passages]

        # 2) Embedding texte
        text_embs = embed.text(
            texts=docs_text,
            model=self.text_model,
            task_type="search_document"
        )
        text_tensor = torch.tensor(text_embs, dtype=torch.float32, device=self.device)

        # 3) Embedding image
        # nomic.embed.image accepte les PIL.Image
        image_embs = embed.image(
            images=passages,
            model=self.image_model
        )
        image_tensor = torch.tensor(image_embs, dtype=torch.float32, device=self.device)

        # On renvoie les deux matrices d'embeddings
        return [text_tensor, image_tensor]

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        # récupère les deux embeddings de passages
        text_tensor, image_tensor = passage_embeddings
        # calcul des similarités dot-product
        # query_embeddings : [nq, d], text_tensor/image_tensor : [nd, d]
        scores_text  = torch.matmul(query_embeddings, text_tensor.T)
        scores_image = torch.matmul(query_embeddings, image_tensor.T)
        # fusion
        scores = self.alpha * scores_text + (1.0 - self.alpha) * scores_image
        return scores
