from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "amazon_books.csv"
RANDOM_STATE = 42


def find_data_path(filename: str = DATA_FILE) -> Path:
    """Find dataset path relative to current file or working directory."""
    local_candidate = Path(__file__).resolve().parent.parent / "dataset" / filename
    if local_candidate.exists():
        return local_candidate

    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        candidate = base / "dataset" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} under any parent dataset/ folder.")


def normalize_text(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: object) -> List[str]:
    cleaned = normalize_text(text)
    return [token for token in cleaned.split() if len(token) > 2 and token not in ENGLISH_STOP_WORDS]


def load_books(data_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(data_path) if data_path else find_data_path()
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["title", "author", "category", "description"]).copy()
    df["description"] = df["description"].fillna("")
    df["author"] = df["author"].fillna("")
    df["content"] = (df["title"].fillna("") + " " + df["author"] + " " + df["description"]).map(normalize_text)
    return df


def choose_example_titles(df: pd.DataFrame, count: int = 2) -> List[str]:
    examples: List[str] = []
    for _, group in df.groupby("category", dropna=True):
        if not group.empty:
            examples.append(group.iloc[0]["title"])
        if len(examples) >= count:
            break
    if len(examples) < count:
        examples.extend(df["title"].drop_duplicates().tolist()[: count - len(examples)])
    return examples[:count]


class BaseBookRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        self.indices = pd.Series(self.df.index, index=self.df["title"]).drop_duplicates()

    def _similarities_for_index(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    def get_recommendations(self, title: str, num_recommendations: int = 5) -> pd.DataFrame | str:
        if title not in self.indices:
            return f"Book '{title}' not found in the dataset."

        idx = int(self.indices[title])
        similarities = self._similarities_for_index(idx)
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_indices = [i for i in ranked_indices if i != idx][:num_recommendations]
        recommendations = self.df.iloc[ranked_indices][["title", "author", "category"]].copy()
        recommendations["similarity_score"] = similarities[ranked_indices]
        return recommendations


class CosineSimilarityRecommender(BaseBookRecommender):
    def __init__(self, df: pd.DataFrame):
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, max_features=7000)
        self.tfidf_matrix = self.vectorizer.fit_transform(df["content"])
        super().__init__(df)

    def _similarities_for_index(self, idx: int) -> np.ndarray:
        return cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()


class Word2VecRecommender(BaseBookRecommender):
    """Train a lightweight Word2Vec model and average token vectors per book."""

    def __init__(
        self,
        df: pd.DataFrame,
        vector_size: int = 64,
        window: int = 5,
        epochs: int = 8,
        max_training_docs: int = 8000,
    ):
        np.random.seed(RANDOM_STATE)
        self.tokenized_content = df["content"].map(tokenize).tolist()
        training_corpus = self.tokenized_content
        if len(training_corpus) > max_training_docs:
            sampled_idx = np.random.choice(len(training_corpus), size=max_training_docs, replace=False)
            training_corpus = [training_corpus[i] for i in sampled_idx]

        self.model = Word2Vec(
            sentences=training_corpus,
            vector_size=vector_size,
            window=window,
            min_count=1,
            workers=1,
            sg=1,
            seed=RANDOM_STATE,
            epochs=epochs,
        )
        self.doc_vectors = np.vstack([self._document_vector(tokens) for tokens in self.tokenized_content])
        self.doc_norms = np.linalg.norm(self.doc_vectors, axis=1)
        super().__init__(df)

    def _document_vector(self, tokens: List[str]) -> np.ndarray:
        vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
        if not vectors:
            return np.zeros(self.model.vector_size, dtype=np.float32)
        return np.mean(vectors, axis=0)

    def _similarities_for_index(self, idx: int) -> np.ndarray:
        query_vector = self.doc_vectors[idx]
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(len(self.df), dtype=np.float32)
        denom = self.doc_norms * query_norm
        denom = np.where(denom == 0, 1e-12, denom)
        return (self.doc_vectors @ query_vector) / denom


class BertEmbeddingRecommender(BaseBookRecommender):
    """BERT recommender with automatic offline fallback when weights are unavailable."""

    def __init__(self, df: pd.DataFrame, model_name: str = "paraphrase-MiniLM-L3-v2", allow_download: bool = False):
        self.backend = "bert"
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.fallback_vectorizer = None
        self.fallback_matrix = None

        try:
            self.model = SentenceTransformer(model_name, local_files_only=not allow_download)
            self.embeddings = self.model.encode(
                df["content"].tolist(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        except Exception as exc:
            warnings.warn(
                f"BERT model '{model_name}' unavailable ({exc}). Falling back to TF-IDF embedding baseline.",
                RuntimeWarning,
            )
            self.backend = "tfidf_fallback"
            self.fallback_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=12000)
            self.fallback_matrix = self.fallback_vectorizer.fit_transform(df["content"])

        super().__init__(df)

    def _similarities_for_index(self, idx: int) -> np.ndarray:
        if self.backend == "bert":
            return np.dot(self.embeddings, self.embeddings[idx])
        return cosine_similarity(self.fallback_matrix[idx], self.fallback_matrix).flatten()


class PrecisionRecallEvaluator:
    def __init__(self, recommender: BaseBookRecommender, df: pd.DataFrame):
        self.recommender = recommender
        self.df = df
        self.results: Dict[str, dict] = {}

    def evaluate_category(self, category: str, num_queries: int = 3, num_recommendations: int = 5) -> Optional[dict]:
        category_books = self.df[self.df["category"] == category]["title"].drop_duplicates().tolist()
        if len(category_books) < 2:
            return None

        query_count = min(num_queries, len(category_books))
        query_books = np.random.choice(category_books, size=query_count, replace=False)
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        f1_scores: List[float] = []
        examples: List[dict] = []

        for query_book in query_books:
            recommendations = self.recommender.get_recommendations(query_book, num_recommendations)
            if isinstance(recommendations, str) or recommendations.empty:
                continue

            recommended_titles = recommendations["title"].tolist()
            same_category_books = [book for book in category_books if book != query_book]
            hits = sum(1 for title in recommended_titles if title in same_category_books)

            precision = hits / len(recommended_titles) if recommended_titles else 0.0
            recall = hits / len(same_category_books) if same_category_books else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            examples.append(
                {
                    "query_book": query_book,
                    "recommendations": recommendations,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "hits": hits,
                    "total_recommendations": len(recommended_titles),
                }
            )

        if not examples:
            return None

        return {
            "category": category,
            "avg_precision": float(np.mean(precision_scores)),
            "avg_recall": float(np.mean(recall_scores)),
            "avg_f1": float(np.mean(f1_scores)),
            "examples": examples,
        }

    def evaluate_all_categories(self, num_queries: int = 3, num_recommendations: int = 5) -> Dict[str, dict]:
        self.results = {}
        for category in sorted(self.df["category"].dropna().unique()):
            result = self.evaluate_category(category, num_queries=num_queries, num_recommendations=num_recommendations)
            if result:
                self.results[category] = result
        return self.results


def print_examples(title: str, recommender: BaseBookRecommender, example_titles: List[str], num_recommendations: int = 5) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for query_title in example_titles:
        recommendations = recommender.get_recommendations(query_title, num_recommendations=num_recommendations)
        print(f"\nQuery: {query_title}")
        if isinstance(recommendations, str):
            print(recommendations)
            continue
        print(recommendations.to_string(index=False))


# Backward-compatible alias for existing imports.
class ContentBasedRecommender(CosineSimilarityRecommender):
    pass
