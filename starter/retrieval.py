from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from starter.intent import COLORS, MATERIALS, USE_CASE_WORDS, ShoppingIntent
from starter.ollama_embeddings import EmbeddingCache, OllamaEmbeddingClient


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "what", "matters", "have", "preference", "those", "options", "not", "quite",
    "right", "yet", "ask", "about", "one", "specific", "attribute", "still",
    "exploring", "additional", "your", "judgment", "use", "item", "product",
}

# A small retail lexicon handles common wording gaps before the catalog-driven
# feedback step has enough candidates to learn from.
QUERY_SYNONYMS = {
    "footwear": ("shoe", "shoes", "boot", "boots"),
    "waterproof": ("waterproof", "water", "resistant"),
    "jewellery": ("jewelry",),
    "trainers": ("sneakers", "shoes"),
    "handbag": ("purse", "bag"),
}


def text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(value: str) -> list[str]:
    tokens = (
        token.lower()
        for token in TOKEN_RE.findall(value)
        if len(token) > 1 and token.lower() not in STOPWORDS
    )
    return list(dict.fromkeys(tokens))


def parse_price(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


@dataclass(frozen=True)
class ProductRecord:
    parent_asin: str
    title: str
    corpus: str
    semantic_text: str
    price: float | None
    ranking_terms: tuple[str, ...]
    attributes: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class RetrievalEvidence:
    """Ranked stage snapshots used by the evaluator's failure audit."""

    bm25_ranked: tuple[str, ...]
    post_nomic_ranked: tuple[str, ...]
    final_candidates: tuple[str, ...]
    recommended: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "bm25_ranked": list(self.bm25_ranked),
            "post_nomic_ranked": list(self.post_nomic_ranked),
            "final_candidates": list(self.final_candidates),
            "recommended": list(self.recommended),
        }


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One explicit return type for recommendations, planning, and diagnostics."""

    recommendations: tuple[dict[str, str], ...]
    candidate_ids: tuple[str, ...]
    evidence: RetrievalEvidence


class CatalogIndex:
    """Grounded lexical retrieval with local neural semantic reranking."""

    def __init__(
        self,
        catalog_path: str | Path,
        embedder: OllamaEmbeddingClient,
        semantic_weight: float = 0.18,
        rerank_limit: int = 16,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.products: dict[str, ProductRecord] = {}
        self.embedder = embedder
        self.semantic_weight = semantic_weight
        self.rerank_limit = rerank_limit
        self.embedding_cache = embedding_cache
        self._embedding_cache: dict[str, tuple[float, ...]] = {}
        self._build(Path(catalog_path))

    def _build(self, catalog_path: Path) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                fields = {
                    name: text(product.get(name))
                    for name in (
                        "title", "categories", "features", "details", "store", "description"
                    )
                }
                corpus = " ".join(fields.values()).lower()
                self.products[parent_asin] = ProductRecord(
                    parent_asin=parent_asin,
                    title=fields["title"],
                    corpus=corpus,
                    semantic_text=self._semantic_text(fields),
                    price=parse_price(product.get("price")),
                    ranking_terms=tuple(terms(" ".join((
                        fields["title"], fields["categories"], fields["features"]
                    )))[:50]),
                    attributes=self._attributes(product, fields, corpus),
                )
                batch.append((
                    parent_asin,
                    fields["title"],
                    fields["categories"],
                    fields["features"],
                    fields["details"],
                    fields["store"],
                    fields["description"],
                ))
                if len(batch) == 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()

        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    @staticmethod
    def _semantic_text(fields: dict[str, str]) -> str:
        # Short, labelled text gives the embedding model the useful product
        # facts without feeding it a long block of repeated catalog metadata.
        parts = [
            f"title: {fields['title']}",
            f"category: {fields['categories']}",
            f"features: {fields['features']}",
            f"details: {fields['details']}",
            f"description: {fields['description']}",
        ]
        return " ".join(part for part in parts if part.split(": ", 1)[1])[:1400]

    def _semantic_rerank(self, query: str, candidate_ids: list[str]) -> list[str]:
        candidates = candidate_ids[:self.rerank_limit]
        if self.embedding_cache:
            cached = self.embedding_cache.get_many([
                item for item in candidates if item not in self._embedding_cache
            ])
            self._embedding_cache.update(cached)
        missing = [item for item in candidates if item not in self._embedding_cache]
        inputs = [f"search_query: {query}"]
        inputs.extend(
            f"search_document: {self.products[item].semantic_text}"
            for item in missing
        )
        vectors = self.embedder.embed(inputs)
        query_vector = vectors[0]
        new_vectors = dict(zip(missing, vectors[1:]))
        self._embedding_cache.update(new_vectors)
        if self.embedding_cache and new_vectors:
            self.embedding_cache.put_many(new_vectors)
        semantic_order = sorted(
            candidates,
            key=lambda item: sum(
                left * right
                for left, right in zip(query_vector, self._embedding_cache[item])
            ),
            reverse=True,
        )
        semantic_rank = {item: rank for rank, item in enumerate(semantic_order, start=1)}

        # Reciprocal-rank fusion is stable across different score ranges. A
        # small semantic weight improves meaning matches without throwing away
        # the strong exact-match ordering from BM25.
        fused = {
            item: 1.0 / (60 + lexical_rank)
            + self.semantic_weight / (60 + semantic_rank[item])
            for lexical_rank, item in enumerate(candidates, start=1)
        }
        reranked = sorted(candidates, key=fused.get, reverse=True)
        return [*reranked, *candidate_ids[self.rerank_limit:]]

    @staticmethod
    def _attributes(product: dict, fields: dict[str, str], corpus: str) -> dict[str, tuple[str, ...]]:
        corpus_terms = set(terms(corpus))
        categories = product.get("categories") or []
        details = product.get("details") or {}
        features = product.get("features") or []

        attributes: dict[str, tuple[str, ...]] = {}
        if corpus_terms & MATERIALS:
            attributes["material"] = tuple(sorted(corpus_terms & MATERIALS))
        if corpus_terms & COLORS:
            attributes["color"] = tuple(sorted(corpus_terms & COLORS))
        if corpus_terms & USE_CASE_WORDS:
            attributes["use_case"] = tuple(sorted(corpus_terms & USE_CASE_WORDS))
        if categories:
            attributes["category"] = (str(categories[-1]).lower(),)
        if fields["store"]:
            attributes["brand"] = (fields["store"].lower()[:80],)
        if features:
            cleaned_features = tuple(
                value for value in (" ".join(str(item).lower().split())[:100] for item in features)
                if value
            )
            if cleaned_features:
                attributes["feature"] = cleaned_features[:4]
        if isinstance(details, dict):
            for key, value in details.items():
                lowered_key = str(key).lower()
                cleaned_value = " ".join(str(value).lower().split())[:80]
                if not cleaned_value:
                    continue
                if any(word in lowered_key for word in ("size", "width")):
                    attributes.setdefault("size", tuple())
                    attributes["size"] = (*attributes["size"], cleaned_value)
                if any(word in lowered_key for word in ("style", "fit", "sleeve", "neck", "pattern", "department")):
                    attributes.setdefault("style", tuple())
                    attributes["style"] = (*attributes["style"], cleaned_value)
        price = parse_price(product.get("price"))
        if price is not None:
            price_bucket = int(price // 25) * 25
            attributes["budget"] = (f"{price_bucket}-{price_bucket + 24}",)
        return attributes

    def _rank_query(self, query: str, limit: int, expand: bool = True) -> list[str]:
        query_terms = terms(query)[:40]
        if not query_terms:
            return []
        if expand:
            expanded_terms = list(query_terms)
            for term in query_terms:
                expanded_terms.extend(QUERY_SYNONYMS.get(term, ()))
            query_terms = list(dict.fromkeys(expanded_terms))[:40]
        expression = " OR ".join(f'"{term}"' for term in query_terms)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _feedback_query(self, base_ids: list[str], original_query: str) -> str:
        # This is a lightweight form of distributional semantics: terms that
        # repeatedly occur near the first-pass results expand a vague query.
        original_terms = set(terms(original_query))
        frequency: Counter[str] = Counter()
        for parent_asin in base_ids[:16]:
            frequency.update(set(self.products[parent_asin].ranking_terms))

        feedback_terms = [
            term
            for term, count in frequency.most_common(30)
            if term not in original_terms and 2 <= count <= 12
        ][:5]
        return " ".join([original_query, *feedback_terms]).strip()

    def _valid_for_constraints(self, product: ProductRecord, intent: ShoppingIntent) -> bool:
        for exclusion in intent.exclusions:
            exclusion_terms = terms(exclusion)
            if exclusion_terms and all(
                re.search(rf"\b{re.escape(term)}\b", product.corpus)
                for term in exclusion_terms
            ):
                return False
        if intent.budget_min is not None and product.price is not None:
            if product.price < intent.budget_min:
                return False
        if intent.budget_max is not None and product.price is not None:
            if product.price > intent.budget_max:
                return False
        return True

    def search(
        self,
        intent: ShoppingIntent,
        seen_products: set[str],
        top_k: int,
        conversation_query: str,
        use_semantic_reranker: bool = True,
    ) -> RetrievalResult:
        candidate_limit = min(300, max(100, top_k * 12 + len(seen_products)))
        full_query = intent.query_text() or intent.category
        lexical_limit = min(300, max(100, top_k + len(seen_products)))
        lexical_ranked = self._rank_query(conversation_query, lexical_limit, expand=False)
        bm25_ranked = list(lexical_ranked)
        if lexical_ranked and use_semantic_reranker:
            lexical_ranked = self._semantic_rerank(conversation_query, lexical_ranked)
        post_nomic_ranked = list(lexical_ranked)
        recommendation_ids = [
            parent_asin for parent_asin in lexical_ranked
            if parent_asin not in seen_products
        ][:top_k]

        # Most queries have a strong lexical match. Only pay for multiple local
        # retrieval routes when the first pass cannot fill the result slate.
        semantic_candidates: list[str] = []
        if len(recommendation_ids) < top_k:
            routes: list[tuple[str, float]] = [(full_query, 2.0)]
            if intent.category and intent.category != full_query:
                routes.append((intent.category, 1.0))
            for value in intent.values()[-4:]:
                routes.append((f"{intent.category} {value}", 1.25))

            first_pass = self._rank_query(full_query, candidate_limit)
            if intent.route == "browsing" and first_pass:
                feedback = self._feedback_query(first_pass, full_query)
                if feedback != full_query:
                    routes.append((feedback, 0.55))

            fused: dict[str, float] = {}
            for query, weight in routes:
                for rank, parent_asin in enumerate(self._rank_query(query, candidate_limit), start=1):
                    fused[parent_asin] = fused.get(parent_asin, 0.0) + weight / (60 + rank)
            semantic_candidates = sorted(fused, key=fused.get, reverse=True)
            for parent_asin in semantic_candidates:
                if len(recommendation_ids) >= top_k:
                    break
                if (
                    parent_asin not in seen_products
                    and parent_asin not in recommendation_ids
                    and self._valid_for_constraints(self.products[parent_asin], intent)
                ):
                    recommendation_ids.append(parent_asin)

        recommendations = [
            {"parent_asin": parent_asin}
            for parent_asin in recommendation_ids
        ]
        question_candidates = list(dict.fromkeys([*lexical_ranked, *semantic_candidates]))
        # The index reports stages, but it never receives the correct target.
        # This keeps diagnostic evidence separate from recommendation logic.
        evidence = RetrievalEvidence(
            bm25_ranked=tuple(bm25_ranked),
            post_nomic_ranked=tuple(post_nomic_ranked),
            final_candidates=tuple(question_candidates[:100]),
            recommended=tuple(recommendation_ids),
        )
        return RetrievalResult(
            recommendations=tuple(recommendations),
            candidate_ids=tuple(question_candidates[:100]),
            evidence=evidence,
        )
