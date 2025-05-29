import math
from typing import List, Dict, Optional, Tuple, Any

import pandas as pd
import torch
from torch import Tensor
from PIL import Image
from tqdm import tqdm


class MMDocIRBench:
    """
    Benchmark for MMDocIR page-level and layout-level retrieval,
    with support for sub-tests and intermediate result inspection.

    Expects:
      - pages_parquet: path to MMDocIR_pages.parquet
      - layouts_parquet: path to MMDocIR_layouts.parquet
      - ann_jsonl:       path to MMDocIR_annotations.jsonl
    """
    def __init__(
        self,
        pages_parquet: str,
        layouts_parquet: str,
        ann_jsonl: str,
    ):
        # Load the raw files
        self.pages_df = pd.read_parquet(pages_parquet)
        self.layouts_df = pd.read_parquet(layouts_parquet)
        self.ann = pd.read_json(ann_jsonl, lines=True)

        # Build flat list of all questions across documents
        self.queries: List[str] = []
        self.query_ids: List[int] = []
        self.qrels_page: Dict[int, List[int]] = {}
        self.qrels_layout: Dict[int, List[int]] = {}

        qid_counter = 0
        # A mapping to find layout_id by (page_id, bbox) tuple
        layout_index = {}
        for _, row in self.layouts_df.iterrows():
            key = (row["page_id"], tuple(row["bbox"]))
            layout_index.setdefault(key, []).append(int(row["layout_id"]))

        # Flatten annotations
        for doc in self.ann.itertuples():
            for qa in doc.questions:
                qid = qid_counter
                self.query_ids.append(qid)
                self.queries.append(qa["Q"])

                # Page-level qrels: map each page_id to its integer index in pages_df
                ppt_indices = []
                for pid in qa["page_id"]:
                    # assume pages_df.passage_id matches pid
                    idx = self.pages_df.index[self.pages_df["passage_id"] == pid].tolist()
                    ppt_indices.extend(idx)
                self.qrels_page[qid] = ppt_indices

                # Layout-level qrels: map each mapping box to layout_id(s)
                lay_indices = []
                for m in qa["layout_mapping"]:
                    key = (m["page"], tuple(m["bbox"]))
                    lay_indices.extend(layout_index.get(key, []))
                self.qrels_layout[qid] = lay_indices

                qid_counter += 1

    def _mrr_at_k(self, retrieved: List[List[int]], qrels: Dict[int, List[int]], k: int) -> float:
        rr_sum = 0.0
        nq = len(retrieved)
        for i, docs in enumerate(retrieved):
            rr = 0.0
            for rank, d in enumerate(docs[:k], start=1):
                if d in qrels.get(i, []):
                    rr = 1.0 / rank
                    break
            rr_sum += rr
        return rr_sum / nq

    def _map_at_k(self, retrieved: List[List[int]], qrels: Dict[int, List[int]], k: int) -> float:
        ap_sum = 0.0
        nq = len(retrieved)
        for i, docs in enumerate(retrieved):
            rels = qrels.get(i, [])
            num_rel = 0
            score = 0.0
            for idx, d in enumerate(docs[:k], start=1):
                if d in rels:
                    num_rel += 1
                    score += num_rel / idx
            if rels:
                ap_sum += score / min(len(rels), k)
        return ap_sum / nq

    def _ndcg_at_k(self, retrieved: List[List[int]], qrels: Dict[int, List[int]], k: int) -> float:
        ndcg_sum = 0.0
        nq = len(retrieved)
        for i, docs in enumerate(retrieved):
            rels = set(qrels.get(i, []))
            dcg = 0.0
            for rank, d in enumerate(docs[:k], start=1):
                if d in rels:
                    dcg += 1.0 / math.log2(rank + 1)
            ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(rels), k) + 1))
            ndcg_sum += (dcg / ideal) if ideal > 0 else 0.0
        return ndcg_sum / nq

    def evaluate(
        self,
        retriever,
        level: str = "page",
        batch_size_encode: int = 8,
        corpus_batch_size: int = 16,
        top_k: List[int] = [1, 3, 5, 10, 20, 50, 100],
        sub_qids: Optional[List[int]] = None,
        return_intermediate: bool = False,
    ) -> Any:
        """
        Run evaluation.

        Args:
          retriever: implements
            - forward_passages(passages: List[Image], batch_size: int) -> List[Tensor]
            - forward_queries(queries: List[str], batch_size: int) -> Tensor
            - get_scores(Q: Tensor, D: List[Tensor]) -> Tensor (n_q, n_d)
          level: "page" or "layout"
          sub_qids: optional list of query-IDs to restrict evaluation
          return_intermediate: if True, also return per-query retrieved lists

        Returns:
          metrics dict, or (metrics, intermediate) if return_intermediate.
        """
        # select appropriate qrels and corpus
        if level == "page":
            qrels = self.qrels_page
            df = self.pages_df
            get_img = lambda row: Image.open(row["image_path"])
        elif level == "layout":
            qrels = self.qrels_layout
            df = self.layouts_df
            get_img = lambda row: Image.open(row["image_path"])
        else:
            raise ValueError("level must be 'page' or 'layout'")

        # select queries
        if sub_qids is not None:
            qmask = [i in sub_qids for i in self.query_ids]
            queries = [q for q, keep in zip(self.queries, qmask) if keep]
            qid_list = [qid for qid, keep in zip(self.query_ids, qmask) if keep]
        else:
            queries = self.queries
            qid_list = self.query_ids

        # encode corpus items
        docs_embs = []
        for start in tqdm(range(0, len(df), corpus_batch_size), desc="Encode corpus"):
            chunk = df.iloc[start : start + corpus_batch_size]
            imgs = [get_img(row) for _, row in chunk.iterrows()]
            docs_embs.extend(retriever.forward_passages(imgs, batch_size_encode))

        # encode queries
        q_emb = retriever.forward_queries(queries, batch_size_encode)

        # score & retrieve
        scores = retriever.get_scores(q_emb, docs_embs)  # (n_q, n_d)
        max_k = max(top_k)
        topk_ids = torch.topk(scores, k=max_k, dim=1).indices.cpu().tolist()

        # metrics
        metrics: Dict[str, float] = {}
        intermediate: Dict[int, Dict[str, Any]] = {}

        for k in sorted(top_k):
            retrieved_at_k = [ids[:k] for ids in topk_ids]
            metrics[f"MRR@{k}"] = self._mrr_at_k(retrieved_at_k, qrels, k)
            metrics[f"MAP@{k}"] = self._map_at_k(retrieved_at_k, qrels, k)
            metrics[f"nDCG@{k}"] = self._ndcg_at_k(retrieved_at_k, qrels, k)

            if return_intermediate:
                for idx, qid in enumerate(qid_list):
                    intermediate.setdefault(qid, {
                        "relevant": qrels.get(qid, []),
                        "retrieved": {}
                    })
                    intermediate[qid]["retrieved"][k] = retrieved_at_k[idx]

        if return_intermediate:
            return metrics, intermediate
        return metrics

    def evaluate_subtest(self, retriever, sub_qids: List[int], **kwargs):
        """Run a sub-test on only a subset of queries, returning intermediate results."""
        return self.evaluate(
            retriever,
            sub_qids=sub_qids,
            return_intermediate=True,
            **kwargs
        )
bench = MMDocIRBench(
    pages_parquet="path/to/MMDocIR_pages.parquet",
    layouts_parquet="path/to/MMDocIR_layouts.parquet",
    ann_jsonl="path/to/MMDocIR_annotations.jsonl"
)

# Full page-level eval:
metrics = bench.evaluate(retriever, level="page")
print(metrics)

# Layout-level eval with intermediate per-query results:
metrics, inter = bench.evaluate(
    retriever,
    level="layout",
    top_k=[1,5,10],
    return_intermediate=True
)
print(metrics)
print(inter[42])  # see query #42's retrieved vs relevant layouts
