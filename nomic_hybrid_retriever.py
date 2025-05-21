from __future__ import annotations
import math
from typing import List, Optional, Union, cast
import torch
from PIL import Image
import pytesseract
from nomic import embed
from tqdm import tqdm
from torch import Tensor
import torch.nn.functional as F
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor 

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
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)
        self.alpha = alpha

        self.image_processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        self.vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5",trust_remote_code = True)

        self.tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
        self.text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5',trust_remote_code=True)
        
        self.device = get_torch_device(device)

    def forward_queries(
        self,
        queries: List[str],
        batch_size: int,
        **kwargs,
    ) -> torch.Tensor:
        
        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)

            query_texts = ["search_query: " + query for query in query_batch]
            encoded_input = self.tokenizer(
                query_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                qs = self.text_model(**encoded_input)
                qs = self._mean_pooling(qs, encoded_input["attention_mask"])
                qs = F.layer_norm(qs, normalized_shape=(qs.shape[1],))
                qs = F.normalize(qs, p=2, dim=1)

            query_embeddings = torch.tensor(qs).to(self.device)
            list_emb_queries.extend(list(torch.unbind(query_embeddings, dim=0)))

        return list_emb_queries

    @staticmethod
    def _mean_pooling(model_output: Tensor, attention_mask: Tensor) -> Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward_passages(
        self,
        passages: List[Image.Image],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
         
