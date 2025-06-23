import os
import yaml
import redis
from falkordb import FalkorDB
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict
import numpy as np
from . import config



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
                    tokenized_doc = lt.lemmatize_text(text_for_table)
                    self.tokenized_corpus.extend(tokenized_doc + columns.split(" "))  # keep original column names too
                    self.table_names_in_order.append(table_name)

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str) -> List[Dict[str, any]]:
        tokenized_query = lt.lemmatize_text(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores for consistent fusion
        max_score = np.max(bm25_scores)
        norm_scores = bm25_scores / max_score if max_score > 0 else bm25_scores

        return [{'table': name, 'score': score}
                for name, score in zip(self.table_names_in_order, norm_scores)]



# --- Retrieval Head 1: Example-Based Retrieval ---
def retrieve_by_example(user_query_embedding: list, graph, top_k: int) -> Dict[str, float]:
    """
    Finds the most similar SampleQuestion nodes and returns the tables they use,
    along with a relevance score.
    """
    print("   - Running Head A: Example-Based Retrieval...")

    # 1. Find top K similar SampleQuestion nodes via vector search
    cypher_query = """
        CALL db.idx.vector.query('SampleQuestion', 'embedding', $k, $embedding)
        YIELD node, score
        // 2. Follow the :USES_TABLE relationship to get the tables
        MATCH (node)-[:USES_TABLE]->(table:Table)
        RETURN table.name AS table_name, (1 - score) AS similarity
    """

    try:
        result = graph.query(cypher_query, {'k': top_k, 'embedding': user_query_embedding}).result_set
    except Exception as e:
        print(f"     - Example-based search failed (is the index created?): {e}")
        return {}

    # 3. Aggregate scores for each table
    # If multiple top questions use the same table, we want to boost its score.
    table_scores = {}
    for record in result:
        table_name, similarity = record[0], record[1]
        if table_name not in table_scores:
            table_scores[table_name] = 0.0
        # We sum the similarities to reward tables that appear in multiple relevant examples
        table_scores[table_name] += similarity

    # 4. Normalize the final scores to a [0, 1] range
    max_score = max(table_scores.values()) if table_scores else 0
    if max_score > 0:
        for table in table_scores:
            table_scores[table] /= max_score

    print(f"     - Found tables via examples: {list(table_scores.keys())}")
    return table_scores


# --- Retrieval Head 2: Schema-Based Retrieval ---
def retrieve_by_schema(user_query: str, user_query_embedding: list, graph, bm25_searcher, top_k: int, alpha: float) -> \
Dict[str, float]:
    """
    Performs a hybrid search (BM25 + HAN embeddings on Tables) to find relevant tables directly.
    This is our previous hybrid retriever, now refactored as a helper.
    """
    print("   - Running Head B: Schema-Based Retrieval...")

    # 1. Dense Search (HAN embeddings on Table nodes)
    dense_cypher = """
        CALL db.idx.vector.query('Table', 'han_embedding', $k, $embedding)
        YIELD node, score
        RETURN node.name AS table_name, (1 - score) AS similarity
    """
    try:
        result = graph.query(dense_cypher, {'k': top_k, 'embedding': user_query_embedding}).result_set
        dense_results = [{'table': r[0], 'score': r[1]} for r in result]
    except Exception as e:
        print(f"     - Schema-based dense search failed: {e}")
        dense_results = []

    # 2. Sparse Search (BM25)
    sparse_results = bm25_searcher.search(user_query)  # Assumes search() returns normalized scores

    # 3. Fuse the results
    combined_scores = {}
    all_tables = set([r['table'] for r in dense_results] + [r['table'] for r in sparse_results])

    dense_map = {r['table']: r['score'] for r in dense_results}
    sparse_map = {r['table']: r['score'] for r in sparse_results}

    for table in all_tables:
        dense_score = dense_map.get(table, 0.0)
        sparse_score = sparse_map.get(table, 0.0)
        combined_scores[table] = (alpha * dense_score) + ((1 - alpha) * sparse_score)

    # 4. Normalize the final scores
    max_score = max(combined_scores.values()) if combined_scores else 0
    if max_score > 0:
        for table in combined_scores:
            combined_scores[table] /= max_score

    print(f"     - Found tables via schema: {list(combined_scores.keys())}")
    return combined_scores


# --- Main Orchestrator Function ---
def find_candidate_tables_advanced(user_query: str, limit: int = 3, alpha_schema: float = 0.5,
                                   alpha_fusion: float = 0.7) -> List[str]:
    """
    Orchestrates the advanced retrieval process by calling both retrieval heads
    and fusing their results for a final, robust list of candidate tables.
    """
    print(f"\n🔎 Advanced retrieval for query: '{user_query}'")

    # --- Setup ---
    r = redis.Redis(host=config.FALKORDB_HOST, port=config.FALKORDB_PORT)
    falkordb = FalkorDB(r)
    graph = falkordb.select_graph(config.GRAPH_NAME)
    text_embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    bm25_searcher = BM25Searcher()  # In a real app, this would be a singleton

    # Embed the user query once
    user_query_embedding = text_embed_model.encode(user_query).tolist()

    # --- 1. Parallel Retrieval ---
    # Call both retrieval heads
    example_scores = retrieve_by_example(user_query_embedding, graph, top_k=5)
    schema_scores = retrieve_by_schema(user_query, user_query_embedding, graph, bm25_searcher, top_k=10,
                                       alpha=alpha_schema)

    # --- 2. Final Fusion ---
    print("\n   - Fusing results from both retrieval heads...")
    final_scores = {}
    all_tables = set(example_scores.keys()) | set(schema_scores.keys())

    for table in all_tables:
        score_from_examples = example_scores.get(table, 0.0)
        score_from_schema = schema_scores.get(table, 0.0)

        # Give more weight to the example-based retrieval as it's more powerful
        final_scores[table] = (alpha_fusion * score_from_examples) + ((1 - alpha_fusion) * score_from_schema)

    # --- 3. Sort and Return ---
    sorted_tables = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

    final_table_list = [table for table, score in sorted_tables[:limit]]

    print(f"\n✅ Final candidate tables after fusion: {final_table_list}")
    return final_table_list

# You would call find_candidate_tables_advanced from your agent.py