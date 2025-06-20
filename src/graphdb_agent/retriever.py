import os
from typing import List, Dict

import numpy as np
import yaml
from falkordb import FalkorDB
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# pip install spacy
# python -m spacy download en_core_web_sm
import spacy
from graphdb_agent import config

# text_to_sql_agent/retriever.py

import os
import yaml
import spacy # <-- Import spaCy
from rank_bm25 import BM25Okapi
# ... other imports

# --- Load spaCy model once ---
# In a real app, this would be part of a class or global setup
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text: str) -> List[str]:
    """
    Processes text using spaCy to lemmatize and remove stop words and punctuation.
    """

    doc = nlp(text.lower())
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return lemmas

# --- One-time Setup for BM25 ---
# This class can be instantiated once and reused.
class BM25Searcher:
    table_names_in_order: list

    def __init__(self):
        print("Initializing BM25 Searcher...")
        self.tokenized_corpus = []
        self.table_names_in_order = []

        # Loop through YAML files to build the corpus for BM25
        for filename in os.listdir(config.SCHEMA_DIR):
            if filename.endswith(".yaml") and filename != "_global_join_paths.yaml":
                yaml_path = os.path.join(config.SCHEMA_DIR, filename)
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    table_name = data['table_name']
                    description = data.get('table_description', '')
                    columns = ' '.join([c['name'] for c in data.get('columns', [])])
                    synonyms = ' '.join([s for c in data.get('columns', []) for s in c.get('synonyms', [])])

                    # Create a comprehensive text document for each table
                    text_for_table = f"{table_name} {description} {columns} {synonyms}"
                    tokenized_doc = lemmatize_text(text_for_table)
                    self.tokenized_corpus.extend(tokenized_doc +  columns.split(" "))  # keep original column names too
                    self.table_names_in_order.append(table_name)

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str) -> List[Dict[str, any]]:
        tokenized_query = lemmatize_text(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores for consistent fusion
        max_score = np.max(bm25_scores)
        norm_scores = bm25_scores / max_score if max_score > 0 else bm25_scores

        return [{'table': name, 'score': score}
                for name, score in zip(self.table_names_in_order, norm_scores)]


def find_candidate_tables_hybrid(user_query: str, limit: int = 3, alpha: float = 0.5):
    """
    Finds candidates using a manual hybrid search with proper normalization.
    """
    print(f"\n🔎 Finding candidate tables for query: '{user_query}'")
    print("   - Using MANUAL similarity calculation (for older FalkorDB versions).")

    falkor_db = FalkorDB(host='localhost', port=6379)
    graph = falkor_db.select_graph(config.GRAPH_NAME)

    text_embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

    # Step 1: Embed the user query
    print("   - Embedding user query...")
    query_embedding = text_embed_model.encode([user_query])  # Encode as a list for sklearn

    # Step 2: Fetch ALL table names and their embeddings from FalkorDB
    print("   - Fetching all table embeddings from the database...")
    cypher_query = """
        MATCH (t:Table)
        WHERE t.han_embedding IS NOT NULL
        RETURN t.name AS table_name, t.han_embedding AS embedding
    """
    result = graph.query(cypher_query).result_set

    if not result:
        print("   - No tables with embeddings found in the database.")
        return []

    # Unpack the results
    table_names = [record[0] for record in result]
    table_embeddings = np.array([record[1] for record in result])

    # calculate raw cosine similarity scores
    raw_cosine_scores = cosine_similarity(query_embedding, table_embeddings)[0]  # Shape: (3,)

    # normalize the cosine scores to [0,1] // TODO: verify the max_cos_score < 0 case
    max_cos_score = np.max(raw_cosine_scores)
    norm_cosine_scores = raw_cosine_scores / max_cos_score if max_cos_score > 0 else raw_cosine_scores

    # --- Semantic Search scoring via Embeddings ---
    dense_results = [{'table': name, 'score': score} for name, score in zip(table_names, norm_cosine_scores)]

    # --- Sparse Search scoring via BM25 ---
    bm25_searcher = BM25Searcher()
    sparse_results = bm25_searcher.search(user_query)

    # --- Fusion of both scores ---
    combined_scores = {}
    for res in dense_results:
        combined_scores[res['table']] = alpha * res['score']

    for res in sparse_results:
        if res['table'] not in combined_scores:
            combined_scores[res['table']] = 0.0
        combined_scores[res['table']] += (1 - alpha) * res['score']

    # Final ranking based on hybrid seach
    sorted_tables = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    final_tables = [table for table, score in sorted_tables[:limit]]

    return final_tables

def find_candidate_tables(user_query: str, limit: int = 3) -> List[str]:
    """
    Finds the most relevant tables for a user query by fetching all embeddings
    and calculating cosine similarity manually in Python.

    This is a fallback for older FalkorDB versions without built-in vector search.
    """
    print(f"\n🔎 Finding candidate tables for query: '{user_query}'")
    print("   - Using MANUAL similarity calculation (for older FalkorDB versions).")

    falkor_db = FalkorDB(host='localhost', port=6379)
    graph = falkor_db.select_graph(config.GRAPH_NAME)

    text_embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

    # Step 1: Embed the user query
    print("   - Embedding user query...")
    query_embedding = text_embed_model.encode([user_query])  # Encode as a list for sklearn

    # Step 2: Fetch ALL table names and their embeddings from FalkorDB
    print("   - Fetching all table embeddings from the database...")
    cypher_query = """
        MATCH (t:Table)
        WHERE t.han_embedding IS NOT NULL
        RETURN t.name AS table_name, t.han_embedding AS embedding
    """
    result = graph.query(cypher_query).result_set

    if not result:
        print("   - No tables with embeddings found in the database.")
        return []

    # Unpack the results
    table_names = [record[0] for record in result]
    table_embeddings = np.array([record[1] for record in result])

    # Step 3: Calculate cosine similarity for all tables at once
    print("   - Calculating similarities in Python...")
    # cosine_similarity expects 2D arrays: (n_samples1, n_features), (n_samples2, n_features)
    similarities = cosine_similarity(query_embedding, table_embeddings)[0]  # Get the first (and only) row

    # Step 4: Combine names with scores and sort
    scored_tables = sorted(zip(table_names, similarities), key=lambda x: x[1], reverse=True)

    # Step 5: Get the top N candidates
    top_candidates = [table_name for table_name, score in scored_tables[:limit]]

    if top_candidates:
        print(f"   - Found candidate tables: {top_candidates}")
    else:
        print("   - No candidate tables found.")

    return top_candidates

# --- Main Execution Block for Demonstration ---
if __name__ == "__main__":


    kk= BM25Searcher()
    kk.tokenized_corpus


    user_query="What is the average order value for customers who signed up in the last quarter?"
    user_query =  "Who is the supplier?"
    retval_emb = find_candidate_tables(user_query,3)
    retval_hybrid = find_candidate_tables_hybrid(user_query,3)

    print(50*"-")
    print(f"{retval_emb=}")
    print(f"{retval_hybrid=}")

