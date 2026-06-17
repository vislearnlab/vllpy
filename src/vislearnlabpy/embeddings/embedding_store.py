from typing import Optional
from docarray import BaseDoc, DocList
from docarray.documents import TextDoc
from docarray.typing import ImageUrl, ImageNdArray, NdArray
from vislearnlabpy.models.clip_model import CLIPGenerator
from vislearnlabpy.embeddings.utils import cleaned_doc_path, normalize_embeddings, indexed_embeddings
from vislearnlabpy.embeddings.similarity_generator import SimilarityGenerator
from docarray.index import InMemoryExactNNIndex
from docarray.utils.filter import filter_docs
from itertools import zip_longest
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import os


# ── Dynamic schema factories ──────────────────────────────────────────────────

def _image_embedding_type(dim: int):
    """Return a BaseDoc subclass for image embeddings with the given dimension."""
    return type(
        f"ImageEmbedding{dim}",
        (BaseDoc,),
        {
            "__annotations__": {
                "embedding": ImageNdArray[dim],
                "url": Optional[ImageUrl],
                "text": Optional[str],
                "normed_embedding": Optional[ImageNdArray[dim]],
            }
        },
    )

def _text_embedding_type(dim: int):
    """Return a TextDoc subclass for text embeddings with the given dimension."""
    return type(
        f"TextEmbedding{dim}",
        (TextDoc,),
        {
            "__annotations__": {
                "embedding": NdArray[dim],
                "normed_embedding": Optional[NdArray[dim]],
            }
        },
    )

# ── Backward-compatible aliases (CLIP = 512) ─────────────────────────────────
CLIPImageEmbedding = _image_embedding_type(512)
CLIPTextEmbedding  = _text_embedding_type(512)


class EmbeddingStore():
    def __init__(self, EmbeddingList=None, FeatureGenerator=None, EmbeddingType=None, dim: int = None):
        # Resolve dim: explicit > inferred from generator > default 512
        if dim is None:
            dim = getattr(FeatureGenerator, "embedding_dim", 512) if FeatureGenerator is not None else 512

        if EmbeddingType is None:
            self.EmbeddingType = _image_embedding_type(dim)
        else:
            self.EmbeddingType = EmbeddingType

        if EmbeddingList is None:
            self.EmbeddingList = DocList[self.EmbeddingType]()
        else:
            self.EmbeddingList = EmbeddingList

        if FeatureGenerator is None:
            self.FeatureGenerator = CLIPGenerator()
        elif hasattr(FeatureGenerator, "model") and hasattr(FeatureGenerator.model, "text_embeddings"):
            # Unwrap EmbeddingGenerator wrappers — store only needs the raw model
            self.FeatureGenerator = FeatureGenerator.model
        else:
            self.FeatureGenerator = FeatureGenerator

    @property
    def embeddings(self):
        return self.EmbeddingList.embedding

    def from_csv(csv_path, feature_generator=None):
        if not os.path.isabs(csv_path):
            csv_path = Path(os.getcwd()) / csv_path
        df = pd.read_csv(csv_path)
        # Detect format: npy paths vs inline numeric columns
        inline_cols = [c for c in df.columns if c.isdigit()]
        if inline_cols:
            # Inline CSV: columns "0", "1", ... hold the embedding values
            dim = len(inline_cols)
            EmbType = _image_embedding_type(dim)
            embedding_doc = DocList[EmbType]()
            for _, row in tqdm(df.iterrows(), total=len(df)):
                try:
                    embedding = np.array([row[c] for c in inline_cols], dtype=np.float32)
                    embedding_doc.append(EmbType(
                        url=row.get("row_id"),
                        embedding=embedding,
                        text=row.get("text"),
                        normed_embedding=None,
                    ))
                except Exception as e:
                    print(f"Failed to load row {row.get('row_id')}: {e}")
        else:
            # npy-path CSV: infer dim from first loadable file
            dim = 512
            for _, row in df.iterrows():
                try:
                    dim = np.load(row["embedding_path"]).shape[-1]
                    break
                except Exception:
                    continue
            EmbType = _image_embedding_type(dim)
            embedding_doc = DocList[EmbType]()
            for _, row in tqdm(df.iterrows(), total=len(df)):
                try:
                    embedding = np.load(row["embedding_path"])
                    embedding_doc.append(EmbType(
                        url=row["row_id"],
                        embedding=embedding,
                        text=row.get("text"),
                        normed_embedding=None,
                    ))
                except Exception as e:
                    print(f"Failed to load embedding for {row['row_id']}: {e}")
        return EmbeddingStore(embedding_doc, feature_generator, EmbeddingType=EmbType, dim=dim)

    def to_base_csv(self, csv_output_path):
        row_data = [
            {"row_id": url, **indexed_embeddings(embedding)}
            for url, embedding in zip(self.EmbeddingList.url, self.EmbeddingList.embedding)
        ]
        pd.DataFrame(row_data).to_csv(csv_output_path, index=False)

    def to_doc(self, doc_output_path):
        self.output_path = cleaned_doc_path(doc_output_path)
        self.EmbeddingList.push(self.output_path)
        return self.output_path

    def from_doc(doc_input_path, dim: int = None, feature_generator=None):
        """Load a DocArray store from disk.

        dim: embedding dimension.  If None, it is inferred from the first document.
             Pass explicitly (e.g. dim=1024) when loading non-CLIP models.
        """
        from docarray.typing import AnyTensor
        if feature_generator is not None and hasattr(feature_generator, "embedding_dim"):
            dim = feature_generator.embedding_dim

        doc_input_path = cleaned_doc_path(doc_input_path)

        if dim is None:
            # Load with a flexible schema to peek at the actual dimension
            FlexDoc = type("FlexDoc", (BaseDoc,), {
                "__annotations__": {
                    "embedding": AnyTensor,
                    "url": Optional[ImageUrl],
                    "text": Optional[str],
                    "normed_embedding": Optional[AnyTensor],
                }
            })
            raw_list = DocList[FlexDoc]().pull(doc_input_path, show_progress=False, local_cache=False)
            dim = int(raw_list[0].embedding.shape[-1]) if len(raw_list) > 0 else 512

        EmbType = _image_embedding_type(dim)
        typed_list = DocList[EmbType]().pull(doc_input_path, show_progress=True, local_cache=False)
        return EmbeddingStore(typed_list, feature_generator, EmbeddingType=EmbType, dim=dim)

    # TODO: add binary save
    def add_embedding(self, embedding, url=None, text=None):
        self.EmbeddingList.append(self.EmbeddingType(
            text=text,
            embedding=embedding,
            url=url,
            normed_embedding=None,
        ))

    def add_embeddings(self, embeddings, urls, texts):
        seen_ids = set()
        temp_list = DocList[self.EmbeddingType](
            [
                self.EmbeddingType(
                    embedding=embedding,
                    url=url if isinstance(url, str) else None,
                    text=text,
                    normed_embedding=None,
                )
                for embedding, url, text in tqdm(zip_longest(embeddings, urls, texts, fillvalue=None))
                if (url if isinstance(url, str) else text) not in seen_ids
                and not seen_ids.add(url if isinstance(url, str) else text)
            ]
        )
        self.EmbeddingList.extend(temp_list)

    def search_store(self, text_query, limit=10, categories=None):
        query = self.FeatureGenerator.text_embeddings([text_query], normalize_embeddings=True)[0].cpu().numpy()
        doc_index = InMemoryExactNNIndex[self.EmbeddingType]()
        if categories is not None:
            filtered_docs = DocList[self.EmbeddingType](filter_docs(self.EmbeddingList, {"text": {"$in": categories}}))
        else:
            filtered_docs = self.EmbeddingList
        filtered_docs.normed_embedding = normalize_embeddings(filtered_docs.embedding)
        doc_index.index(filtered_docs)
        retrieved_docs, scores = doc_index.find(query, search_field="normed_embedding", limit=limit)
        return retrieved_docs, scores

    def retrieve_cross_similarity(self, embedding_list, sim_type="cosine"):
        sim_generator = SimilarityGenerator(similarity_type=sim_type, model=self.FeatureGenerator.model)
        return sim_generator.cross_sims(self.EmbeddingList, embedding_list)

    def retrieve_similarities(self, sim_type="cosine", output_path=None, text_pairs=None,
                              use_urls=False):
        sim_generator = SimilarityGenerator(similarity_type=sim_type, model=self.FeatureGenerator.model)
        if text_pairs is None:
            texts = self.EmbeddingList.text
            urls = self.EmbeddingList.url
            if urls and urls[0] is not None:
                keys = urls
            elif texts:
                keys = texts
            else:
                keys = None
            return sim_generator.all_sims(self.EmbeddingList.embedding, keys, output_path)
        else:
            return sim_generator.specific_sims(self.EmbeddingList, text_pairs, output_path,
                                               use_urls=use_urls)

    def _embeddings_by_url(self) -> dict:
        """Return {url: (embedding, text)} for all URL-keyed embeddings."""
        return {doc.url: (doc.embedding, doc.text)
                for doc in self.EmbeddingList if doc.url is not None}

    def _mean_embeddings_by_text(self) -> dict:
        """Return {text: mean_embedding} for all text-labeled embeddings."""
        groups = {}
        for doc in self.EmbeddingList:
            if doc.text is not None:
                groups.setdefault(doc.text, []).append(doc.embedding)
        return {text: np.mean(embs, axis=0) for text, embs in groups.items()}

    def _resolve_pairs(self, pairs, use_urls, text_labels=None):
        """Yield (embs, text1, ids) for each N-option trial.

        embs: list of N embeddings (one per option)
        text1: target text label for the text embedding lookup
        ids: list of N image IDs
        """
        if use_urls:
            img_by_url = self._embeddings_by_url()
            for i, pair in enumerate(pairs):
                ids = list(pair)
                text1 = text_labels[i] if text_labels is not None else img_by_url.get(ids[0], (None, None))[1]
                if any(id_ not in img_by_url for id_ in ids):
                    continue
                yield [img_by_url[id_][0] for id_ in ids], text1, ids
        else:
            img_by_text = self._mean_embeddings_by_text()
            for i, pair in enumerate(pairs):
                ids = list(pair)
                text1 = text_labels[i] if text_labels is not None else ids[0]
                if any(id_ not in img_by_text for id_ in ids):
                    continue
                yield [img_by_text[id_] for id_ in ids], text1, ids
    # TODO: maybe move to similarity_generator or utils
    def multimodal_prob(self, text_store, pairs, use_urls=False, text_labels=None,
                        logit=None, beta=1.0, rule="softmax"):
        """Softmax or Luce probability P(I_target | T_target) for each N-option trial.

        Works for 2-AFC, 4-AFC, or any N-AFC — pairs contains N image IDs per trial.

        Parameters
        ----------
        text_store : EmbeddingStore
            Text embedding store from the same model.
        pairs : list of tuples
            Each tuple contains N image IDs (URLs if use_urls=True, else text labels).
            The first element is the target option.
        use_urls : bool
            If True, look up images by URL (no averaging). If False, use mean embedding
            per text label.
        text_labels : list of str, optional
            Target text label per trial for the text embedding lookup. If None:
            for URL pairs, inferred from the store's text attribute for the first URL;
            for text pairs, uses the first element of the pair.
        logit : float, optional
            Model's learned temperature applied to raw cosine similarities. Defaults to
            the model's logit_scale (e.g. ~100 for CLIP). Set to 1.0 when passing a
            KL-optimized beta, since that beta already operates on raw cosine sims.
            Ignored when rule="luce".
        beta : float
            Additional exponent on top of logit: softmax(beta * logit * sims).
            Pass the optimal beta from softmax_optimized_kl for human-calibrated scaling.
            For Luce: ((sims + 1) / 2) ** beta. Default 1.0.
        rule : str
            "softmax" (default): softmax(beta * logit * sims).
            "luce": Luce choice rule — sim(I_i, T) / Σ sim(I_j, T). beta and logit are not applied.

        Returns
        -------
        pd.DataFrame with columns: id1..idN, text1, multimodal_prob
        """
        from scipy.special import softmax as scipy_softmax

        txt_by_text = text_store._mean_embeddings_by_text()

        if logit is None:
            model = self.FeatureGenerator
            if hasattr(model, '_resolve_logit_scale'):
                logit = model._resolve_logit_scale()
            elif hasattr(model, 'model') and hasattr(model.model, 'logit_scale'):
                logit = model.model.logit_scale.exp().item()
            else:
                logit = 100.0

        rows = []
        for embs, text1, ids in self._resolve_pairs(pairs, use_urls, text_labels):
            if text1 not in txt_by_text:
                continue
            normed = [e / np.linalg.norm(e) for e in embs]
            t1 = txt_by_text[text1] / np.linalg.norm(txt_by_text[text1])
            sims = np.array([t1 @ e for e in normed])
            if rule == "softmax":
                probs = scipy_softmax(beta * logit * sims)
            elif rule == "luce":
                probs = sims / sims.sum()
            else:
                raise ValueError(f"Unknown rule '{rule}'. Use 'softmax' or 'luce'.")
            row = {f"id{j+1}": id_ for j, id_ in enumerate(ids)}
            row.update({"text1": text1, "multimodal_prob": probs[0]})
            rows.append(row)
        return pd.DataFrame(rows)

    def softmax_optimized_kl(self, text_store, pairs, human_probs, use_urls=False,
                             text_labels=None, beta_bounds=(0.025, 40)):
        """Softmax-optimized KL divergence between human and model response distributions.

        Finds the optimal β* minimizing mean KL(h_t || softmax(β·m_t)) across trials.
        Works for any N-AFC. Matches the R implementation using philentropy::KL + nloptr.

        For 2-AFC, pass human_probs as scalars p ∈ [0,1] (proportion choosing the
        target); [p, 1-p] is constructed internally. For N-AFC, pass arrays of length N.

        Parameters
        ----------
        text_store : EmbeddingStore
            Text embedding store from the same model.
        pairs : list of tuples
            Same format as multimodal_prob.
        human_probs : list of float or array-like
            Per-trial human proportion choosing the target (first) image.
        use_urls : bool
            If True, look up images by URL. If False, use mean embedding per text label.
        text_labels : list of str, optional
            Same as multimodal_prob.
        beta_bounds : (float, float)
            Search range for β (default matches R implementation: 0.025–40).

        Returns
        -------
        dict with keys: beta (optimal exponent), mean_kl (at β*), per_trial_kl (list)
        """
        from scipy.optimize import minimize_scalar
        from scipy.special import softmax as scipy_softmax

        txt_by_text = text_store._mean_embeddings_by_text()

        trial_logits, trial_human = [], []
        for (embs, text1, _), h in zip(self._resolve_pairs(pairs, use_urls, text_labels), human_probs):
            if text1 not in txt_by_text:
                continue
            normed = [e / np.linalg.norm(e) for e in embs]
            t1 = txt_by_text[text1] / np.linalg.norm(txt_by_text[text1])
            trial_logits.append(np.array([t1 @ e for e in normed]))
            h = np.asarray(h, dtype=float)
            trial_human.append(np.array([h, 1 - h]) if h.ndim == 0 else h)

        def mean_kl(beta):
            kls = [np.sum(h * np.log(h / np.clip(scipy_softmax(beta * m), 1e-10, 1.0)))
                   for m, h in zip(trial_logits, trial_human)]
            return np.mean(kls)

        result = minimize_scalar(mean_kl, bounds=beta_bounds, method='bounded')
        beta_opt = result.x
        per_trial_kl = [
            np.sum(h * np.log(h / np.clip(scipy_softmax(beta_opt * m), 1e-10, 1.0)))
            for m, h in zip(trial_logits, trial_human)
        ]
        return {"beta": beta_opt, "mean_kl": result.fun, "per_trial_kl": per_trial_kl}

    def compute_text_rdm(self, sim_type="cosine", output_path=None, order=None, ranked=False):
        from vislearnlabpy.embeddings.similarity_utils import compute_rdm, plot_rdm
        texts = self.EmbeddingList.text
        if texts is None:
            raise ValueError("No text column in embeddings")
         # Unique texts
        unique_texts = sorted(set(texts))

        if order is not None:
            extra = set(order) - set(unique_texts)
            missing = set(unique_texts) - set(order)
            if extra:
                raise ValueError(f"order contains labels not in embeddings: {extra}")
            if missing:
                print(f"Skipping labels in embeddings not in passed in list: {missing}")
            unique_texts = list(order)

        emb_by_text = self._mean_embeddings_by_text()
        text_means = np.stack([emb_by_text[t] for t in unique_texts if t in emb_by_text])
        # Compute RDM
        rdm = compute_rdm(text_means, method=sim_type, ranked=ranked)
        # Plot RDM if output_path is given
        if output_path:
            suffix = "ranked" if ranked else "nonranked"
            plot_rdm(
                out_path=output_path,
                rdm=rdm,
                x_labels=unique_texts,
                y_labels=unique_texts,
                title=f"rdm_{suffix}"
            )
        return rdm
