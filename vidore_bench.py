import io
import math
from typing import List, Dict, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# --- VOTRE RETRIEVER ICI ---
# from votre_module import NomicHybridRetriever

# --- UTILS D'ÉVALUATION ---
def topk_batch(
    q_embs: Tensor,
    d_embs: Tensor,
    top_k: int,
    batch_size: int = 512,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Pour chaque requête, renvoie les top_k indices et scores parmi d_embs,
    en traitant d_embs par tranches de taille batch_size pour limiter la RAM GPU.
    """
    n_q, dim = q_embs.shape
    n_d = d_embs.shape[0]
    # initialisation avec des listes vides
    topk_ids = [torch.empty(0, dtype=torch.long) for _ in range(n_q)]
    topk_scores = [torch.empty(0) for _ in range(n_q)]

    with torch.no_grad():
        for start in range(0, n_d, batch_size):
            end = min(start + batch_size, n_d)
            chunk = d_embs[start:end]  # [D_chunk, dim]
            # scores : [n_q, D_chunk]
            scores = q_embs @ chunk.T
            scores = scores.cpu()
            # pour chaque requête, on merge son précédent topk et ce chunk
            for qi in range(n_q):
                prev_s = topk_scores[qi]
                prev_i = topk_ids[qi]
                cur_s = scores[qi]
                # + on prend les topk dans ce chunk
                vals, idxs = torch.topk(cur_s, k=min(top_k, cur_s.size(0)))
                idxs = idxs + start  # remise à l’échelle globale
                # concaténation
                all_s = torch.cat([prev_s, vals])
                all_i = torch.cat([prev_i, idxs])
                # on reclasse et on garde top_k
                best_s, best_indices = torch.topk(all_s, k=min(top_k, all_s.size(0)))
                best_i = all_i[best_indices]
                topk_scores[qi] = best_s
                topk_ids[qi] = best_i
    # conversion en listes Python
    return [ids.tolist() for ids in topk_ids], [scores.tolist() for scores in topk_scores]


def mrr_at_k(
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
    retriever,
    dataset_name: str,
    split: str = "test",
    batch_size_encode: int = 16,
    batch_size_score: int = 512,
    top_k: int = 100,
    fusion_fn: Callable[[Tensor, Tensor], Tensor] = lambda t, v: t * 0.5 + v * 0.5,
) -> Dict[str, float]:
    """
    Charge corpus/queries/qrels, encode tout en batch, fusionne les scores via fusion_fn,
    puis calcule MRR@K, MAP@K et nDCG@K.
    fusion_fn prend en entrée :
       - T : Tensor [n_q, n_d] (scores texte)
       - V : Tensor [n_q, n_d] (scores vision)
      et doit retourner un Tensor [n_q, n_d].
    """
    # 1. chargement
    ds_c = load_dataset(dataset_name, name="corpus", split=split)
    ds_q = load_dataset(dataset_name, name="queries", split=split)
    ds_r = load_dataset(dataset_name, name="qrels",  split=split)

    # mapping qid -> liste de doc_indices pertinents
    qrels: Dict[int, List[int]] = {}
    for ex in ds_r:
        qid = ex["query_id"]
        docid = ex["doc_id"]
        qrels.setdefault(qid, []).append(docid)

    # 2. encodage des passages (image PIL requise dans "page")
    pages: List[Image.Image] = [
        Image.open(io.BytesIO(ex["image"])) for ex in ds_c
    ]
    text_embs, img_embs = retriever.encode_passages  (  # méthode à implémenter
        passages=pages,
        batch_size=batch_size_encode
    )

    # on empile en deux gros tensors
    D_text = torch.stack(text_embs)  # [n_doc, dim]
    D_vis  = torch.stack(img_embs)   # [n_doc, dim]

    # 3. encodage des requêtes (texte)
    queries: List[str] = [ex["text"] for ex in ds_q]
    Q_text = torch.stack(
        retriever.forward_queries(queries, batch_size=batch_size_encode)
    )  # [n_q, dim]

    # 4. calcul matriciel des scores texte/vision
    # -> [n_q, n_doc]
    with torch.no_grad():
        S_text = Q_text @ D_text.T
        S_vis  = Q_text @ D_vis.T

        # fusion à la volée
        S = fusion_fn(S_text, S_vis)

    # 5. pour chaque requête, on récupère top_k via topk_batch
    topk_ids, _ = topk_batch(S, torch.zeros(0), top_k, batch_size_score)
    # Note : ici on a S complet, donc on peut directement faire
    # ids = torch.topk(S, k=top_k, dim=1).indices.cpu().tolist()

    ids = torch.topk(S, k=top_k, dim=1).indices.cpu().tolist()

    # 6. métriques
    results = {
        "MRR@{}".format(top_k):  mrr_at_k(ids, qrels, top_k),
        "MAP@{}".format(top_k):  average_precision_at_k(ids, qrels, top_k),
        "nDCG@{}".format(top_k): ndcg_at_k(ids, qrels, top_k),
    }
    return results

def encode_passages(
    self,
    passages: List[Image.Image],
    batch_size: int,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Retourne deux listes de tenseurs (texte, vision) pour chaque passage.
    """
    # copier/adapter le code de forward_passages en le scindant
    # pour retourner (text_embs, img_embs) au lieu de les fusionner.


if __name__ == "__main__":
    # exemple d'usage
    retriever = NomicHybridRetriever(alpha=0.6, device="cuda")
    # il faudra y ajouter la méthode encode_passages() qui retourne (text_embs, img_embs)
    metrics = evaluate_retriever(
        retriever,
        dataset_name="vidore/esg_reports_v2",
        split="test",
        batch_size_encode=8,
        batch_size_score=512,
        top_k=50,
        fusion_fn=lambda T, V: 0.7 * T + 0.3 * V,  # liberté totale
    )
    print(metrics)
