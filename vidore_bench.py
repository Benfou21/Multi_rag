import io
import math
from typing import List, Dict, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

class MyVidoreBench(): 


    def mrr_at_k(
        self,
        retrieved: List[List[int]],
        qrels: Dict[int, List[int]],
        k: int
    ) -> float:
        rr_sum = 0.0
        nq = len(retrieved)
        for qi, docs in enumerate(retrieved):
            rr = 0.0
            for rank, d in enumerate(docs[:k], start=1):
                if d in qrels.get(qi, []):
                    rr = 1.0 / rank
                    break
            rr_sum += rr
        return rr_sum / nq


    def average_precision_at_k(
        self,
        retrieved: List[List[int]],
        qrels: Dict[int, List[int]],
        k: int
    ) -> float:
        ap_sum = 0.0
        nq = len(retrieved)
        for qi, docs in enumerate(retrieved):
            rels = qrels.get(qi, [])
            num_rel = 0
            score = 0.0
            for idx, d in enumerate(docs[:k], start=1):
                if d in rels:
                    num_rel += 1
                    score += num_rel / idx
            if len(rels) > 0:
                ap_sum += score / min(len(rels), k)
        return ap_sum / nq


    def ndcg_at_k(
        self,
        retrieved: List[List[int]],
        qrels: Dict[int, List[int]],
        k: int
    ) -> float:
        import math
        ndcg_sum = 0.0
        nq = len(retrieved)
        for qi, docs in enumerate(retrieved):
            rels = set(qrels.get(qi, []))
            dcg = 0.0
            for i, d in enumerate(docs[:k], start=1):
                if d in rels:
                    dcg += 1.0 / math.log2(i + 1)
            # idcg = somme idéale (toutes les vrais docs en tête)
            ideal_rels = min(len(rels), k)
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_rels + 1))
            ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0
        return ndcg_sum / nq


    # --- FONCTION D'ÉVALUATION GLOBALE ---
    def evaluate_retriever(
        self,
        retriever,
        dataset_name: str,
        split: str = "test",
        batch_size_encode: int = 8,
        batch_size_score: int = 512,
        corpus_processing_batch_size: int = 16,
        top_k: List[int] = [1,3,5,10,20,50,100],
    ) -> Dict[str, float]:
        """
        Charge corpus/queries/qrels, encode tout en batch, fusionne les scores via fusion_fn,
        puis calcule MRR@K, MAP@K et nDCG@K.
        fusion_fn prend en entrée :
        - T : Tensor [n_q, n_d] (scores texte)
        - V : Tensor [n_q, n_d] (scores vision)
        et doit retourner un Tensor [n_q, n_d].
        """
        print(f"Loading datasets: {dataset_name} (split: {split})")
        # 1. chargement
        ds_c = load_dataset(dataset_name, name="corpus", split=split)
        ds_q = load_dataset(dataset_name, name="queries", split=split)
        ds_r = load_dataset(dataset_name, name="qrels",  split=split)
        print(f"Loading datasets: Done")

        # mapping qid -> liste de doc_indices pertinents
        qrels: Dict[int, List[int]] = {}
        
        for ex in ds_r:
            qid = ex["query-id"]
            docid = ex["corpus-id"]
            qrels.setdefault(qid, []).append(docid)

        all_passages_embs: List[Tensor] = []
        num_corpus_docs = len(ds_c)

        print(f"\nEncoding pages...")
        for i in tqdm(range(0, num_corpus_docs, corpus_processing_batch_size), desc="Corpus Encoding Batches"):
            batch_start = i
            batch_end = min(i + corpus_processing_batch_size, num_corpus_docs)
            
            current_batch_dataset_slice = ds_c.select(range(batch_start, batch_end))
            current_batch_images: List[Image.Image] = [ex['image'] for ex in current_batch_dataset_slice]


            if not current_batch_images:
                continue

            batch_embs = retriever.forward_passages(
                passages=current_batch_images,
                batch_size=batch_size_encode
            )
            all_passages_embs.extend(batch_embs)

            del current_batch_images, current_batch_dataset_slice, batch_embs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        

        # 3. encodage des requêtes 
        queries: List[str] = [ex["query"] for ex in ds_q]
        queries_emb = retriever.forward_queries(queries, batch_size=batch_size_encode)
        

        # 4. score
        S = retriever.get_scores(queries_emb,all_passages_embs )

        # 5. Top K for multiple K values
        results: Dict[str, float] = {}
        
        # Determine the maximum K to compute topk once
        max_k = max(top_k)

        # Compute topk for the maximum K
        # ids will be (num_queries, max_k)
        all_ids = torch.topk(S, k=max_k, dim=1).indices.cpu().tolist()

        # 6. Calculate metrics for each K
        for k_val in sorted(top_k): # Iterate through sorted k values for consistent output
            # Slice the already computed top-max_k results to get top-k_val results
            current_k_ids = [query_ids[:k_val] for query_ids in all_ids]

            results[f"MRR@{k_val}"] = self.mrr_at_k(current_k_ids, qrels, k_val)
            results[f"MAP@{k_val}"] = self.average_precision_at_k(current_k_ids, qrels, k_val)
            results[f"nDCG@{k_val}"] = self.ndcg_at_k(current_k_ids, qrels, k_val)
        
        del queries_emb, all_passages_embs, S, all_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("\nEvaluation complete.")
        return results
