import os
import yaml
import redis
from falkordb import FalkorDB
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv
from sentence_transformers import SentenceTransformer
from graphdb_agent import config


# --- Update config.py for FalkorDB ---
# You should modify your config.py to have these settings instead of Neo4j's
# FALKORDB_HOST = "localhost"
# FALKORDB_PORT = 6379
# GRAPH_NAME = "text2sql_graph"

class HANModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, metadata, heads=4)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = {key: self.lin(value) for key, value in out.items()}
        return out


def _upload_schema_to_falkordb(graph, schema_data):
    print("--- Uploading schema to FalkorDB ---")

    # --- PHASE 1: Create all nodes and HAS_COLUMN relationships ---
    for table_name, data in schema_data['tables'].items():
        # Update the Table node creation to include all new keys
        table_descriptions = data.get('table_description', [])
        if isinstance(table_descriptions, str):
            table_descriptions = [table_descriptions]

        item_separator = " | "  # define separator for clarity and robustness

        graph.query("""
            MERGE (t:Table {name: $name})
            SET t.description = $desc, 
                t.sample_questions = $questions,
                t.business_rules = $rules,
                t.few_shot_examples = $examples
        """, {
            'name': table_name,
            'desc': " ".join(table_descriptions),
            'questions': item_separator.join(data.get('sample_questions', [])),
            'rules': item_separator.join(data.get('business_rules', [])),
            'examples': item_separator.join(x['question'] for x in data.get('few_shot_examples', []))
        })

        # Update the Column node creation
        for col in data['columns']:
            graph.query("""
                MATCH (t:Table {name: $table_name})
                MERGE (c:Column {name: $col_name, table_name: $table_name})
                MERGE (t)-[:HAS_COLUMN]->(c)
                MERGE (c)-[:BELONGS_TO]->(t)                   
                SET c.description = $desc, 
                    c.type = $type, 
                    c.synonyms = $synonyms,
                    c.rules = $rules               
            """, {
                'table_name': table_name,
                'col_name': col['name'],
                'desc': col.get('description', ''),
                'type': col.get('type', 'TEXT'),
                'synonyms': item_separator.join(col.get('synonyms', [])),
                'rules': item_separator.join(col.get('rules', []))
            })

    # --- PHASE 2: Create LINKS_TO relationships from the global join file ---
    for join in schema_data['join_paths']:
        graph.query("""
             MATCH (t1:Table {name: $from_table}), (t2:Table {name: $to_table})
             MERGE (t1)-[r:LINKS_TO]->(t2)
             SET r.from_column = $from_column, r.to_column = $to_column
         """, {
            'from_table': join['from_table'],
            'to_table': join['to_table'],
            'from_column': join['from_column'],
            'to_column': join['to_column']
        })
    print("Schema upload complete.")


def _run_han_pipeline_falkordb(graph, text_embedding_model):
    print("--- Starting HAN Embedding Pipeline for FalkorDB ---")

    # Step 1: Extract graph data from FalkorDB
    print("Extracting graph data...")

    # FalkorDB returns data in a list of lists, so we map it to dicts for consistency
    print("Extracting graph data with all metadata...")
    tables_res = graph.query("""
        MATCH (t:Table) 
        RETURN id(t) AS id, t.name AS name, t.description AS desc, 
               t.sample_questions AS questions, t.business_rules AS rules, 
               t.few_shot_examples AS examples
    """).result_set
    tables_data = [{'id': r[0], 'name': r[1], 'desc': r[2], 'questions': r[3], 'rules': r[4], 'examples': r[5]} for r in tables_res]

    cols_res = graph.query("""
        MATCH (c:Column) 
        RETURN id(c) AS id, c.name AS name, c.description AS desc, 
               c.type AS type, c.synonyms AS synonyms, c.rules AS rules
    """).result_set
    cols_data = [{'id': r[0], 'name': r[1], 'desc': r[2], 'type': r[3], 'synonyms': r[4], 'rules': r[5]} for r in cols_res]

    # get the forward edges
    has_col_res = graph.query("MATCH (t:Table)-[:HAS_COLUMN]->(c:Column) RETURN id(t) AS src, id(c) AS dest").result_set
    has_col_data = [{'src': r[0], 'dest': r[1]} for r in has_col_res]

    # get the inverse edges
    belongs_to_res = graph.query(
        "MATCH (c:Column)-[:BELONGS_TO]->(t:Table) RETURN id(c) AS src, id(t) AS dest").result_set
    belongs_to_data = [{'src': r[0], 'dest': r[1]} for r in belongs_to_res]

    links_to_res = graph.query(
        "MATCH (t1:Table)-[:LINKS_TO]->(t2:Table) RETURN id(t1) AS src, id(t2) AS dest").result_set
    links_to_data = [{'src': r[0], 'dest': r[1]} for r in links_to_res]

    table_id_map = {r['id']: i for i, r in enumerate(tables_data)}
    col_id_map = {r['id']: i for i, r in enumerate(cols_data)}

    # --- Step 2: Prepare HeteroData object with STRUCTURED text features ---
    print("Preparing data for PyTorch Geometric...")
    data = HeteroData()
    if has_col_data:
        data['table', 'has_column', 'column'].edge_index = torch.tensor(
            [[table_id_map[r['src']] for r in has_col_data], [col_id_map[r['dest']] for r in has_col_data]],
            dtype=torch.long)

    if belongs_to_data:
        data['column', 'belongs_to', 'table'].edge_index = torch.tensor(
            [[col_id_map[r['src']] for r in belongs_to_data], [table_id_map[r['dest']] for r in belongs_to_data]],
            dtype=torch.long)

    # Define separators for clarity and robustness
    section_separator = " ; "
    item_separator = " | "

    # --- Build Rich Table Documents ---
    table_texts = []
    for r in tables_data:
        parts = []
        parts.append(f"Table name: {r['name']}")

        # Add multi-line description
        if r.get('desc') and isinstance(r['desc'], list):
            parts.append(f"Description: {' '.join(r['desc'])}")
        elif r.get('desc'):
            parts.append(f"Description: {r['desc']}")

        # Add business rules
        if r.get('rules'):
            parts.append(f"Business Rules: {r['rules']}")

        # Add sample questions
        if r.get('questions'):
            parts.append(f"This table can answer questions like: '{r['questions']}'")

        # Add questions from few-shot examples
        if r.get('examples'):
            example_questions = r.get('examples')
            parts.append(f"It is also used for complex questions such as: '{' '.join(example_questions)}'")

        table_texts.append(section_separator.join(parts))

    data['table'].x = torch.tensor(text_embedding_model.encode(table_texts), dtype=torch.float)

    # --- Build Rich Column Documents ---
    col_texts = []
    for r in cols_data:
        parts = []
        parts.append(f"Column name: {r['name']}.")
        parts.append(f"Data type: {r.get('type', 'TEXT')}.")

        if r.get('desc'):
            parts.append(f"Description: {r['desc']}")

        if r.get('synonyms'):
            parts.append(f"Synonyms: {r['synonyms']}.")

        if r.get('rules'):
            parts.append(f"Rules: {' '.join(r['rules'])}")

        col_texts.append(section_separator.join(parts))

    data['column'].x = torch.tensor(text_embedding_model.encode(col_texts), dtype=torch.float)

    # Use table_id_map for both src and dest
    if links_to_data:  # Only create the edge_index if there are links
        data['table', 'links_to', 'table'].edge_index = torch.tensor(
            [[table_id_map[r['src']] for r in links_to_data], [table_id_map[r['dest']] for r in links_to_data]],
            dtype=torch.long)

    # Check if the forward relationship exists before trying to reverse it
    if ('table', 'has_column', 'column') in data.edge_types:
        # Get the forward edge_index
        src, dest = data['table', 'has_column', 'column'].edge_index
        # Assign the reversed tensor [dest, src] to the new edge type
        data['column', 'belongs_to', 'table'].edge_index = torch.stack([dest, src], dim=0)

    # Step 3: Run data through the HAN model
    print("Running inference with HAN model...")
    # Handle the case where the graph might be empty or have no edges
    if not data.edge_types:
        print("Warning: Graph has no edges. Skipping HAN model inference.")
        # You might want to handle this case by just using the initial text embeddings
        # For now, we'll just exit the function.
        return

    model = HANModel(
        in_channels=-1,
        hidden_channels=config.HAN_HIDDEN_CHANNELS,
        out_channels=config.HAN_OUT_CHANNELS,
        metadata=data.metadata()
    )
    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x_dict, data.edge_index_dict)

    # Step 4: Write final embeddings back to FalkorDB
    print("Writing final HAN embeddings back to FalkorDB...")
    table_embeddings = final_embeddings['table'].tolist()
    idx_to_table_id = {i: r['id'] for r in tables_data for i in [table_id_map[r['id']]]}

    for i, embedding in enumerate(table_embeddings):
        falkordb_id = idx_to_table_id[i]
        graph.query("""
            MATCH (t:Table) WHERE id(t) = $id
            SET t.han_embedding = $embedding
        """, {'id': falkordb_id, 'embedding': embedding})

    print("HAN pipeline complete.")


def get_schema_data():
    schema_data = {"tables": {}, 'join_paths': {}}
    for filename in os.listdir(config.SCHEMA_DIR):
        if filename.endswith(".yaml"):
            with open(os.path.join(config.SCHEMA_DIR, filename), 'r') as f:
                data = yaml.safe_load(f)
                if "_global_join_paths" in filename:
                    schema_data["join_paths"] = data
                elif "_global_sample_questions" in filename:
                    schema_data["sample_questions"] = data

                else:
                    schema_data['tables'][data['table_name']] = data
    return schema_data


def _create_sample_question_nodes(graph, text_embed_model):
    """
    Parses the sample_questions.yaml file, creates SampleQuestion nodes,
    generates their embeddings, and links them to the tables they use.
    """
    print("\n--- Creating SampleQuestion nodes and relationships ---")

    questions_path = os.path.join(config.SCHEMA_DIR, "_global_sample_questions.yaml")
    if not os.path.exists(questions_path):
        print("Warning: sample_questions.yaml not found. Skipping creation of example nodes.")
        return

    with open(questions_path, 'r') as f:
        all_questions = yaml.safe_load(f) or []

    if not all_questions:
        print("sample_questions.yaml is empty. No nodes to create.")
        return

    print(f"Found {len(all_questions)} sample questions to process...")

    for i, item in enumerate(all_questions):
        # Basic validation
        # if not all({'id', 'question', 'sql', 'tables_used', 'tags'} <= item.keys()):
        #     print(f"Warning: Skipping item {i + 1} due to missing required keys.")
        #     continue

        question_text = item['question']

        # 1. Generate the vector embedding for the question text
        embedding = text_embed_model.encode(question_text).tolist()

        # 2. Create the SampleQuestion node in the graph database
        # We use MERGE on the unique 'id' to make this operation idempotent.
        # If you run the script again, it will update existing nodes instead of creating duplicates.
        graph.query("""
            MERGE (q:SampleQuestion {id: $id})
            SET q.question_text = $question_text,
                q.sql_query = $sql_query,
                q.tags = $tags,
                q.embedding = $embedding
        """, {
            'id': item['id'],
            'question_text': question_text,
            'sql_query': item['sql'],
            'tags': ", ".join(item.get('tags', [])),
            'embedding': embedding
        })

        # 3. Create the :USES_TABLE relationships to link the question to its tables
        for table_name in item['tables_used']:
            # This query finds the question node we just created/updated and the existing table node,
            # then creates a relationship between them.
            graph.query("""
                MATCH (q:SampleQuestion {id: $qid})
                MATCH (t:Table {name: $tname})
                MERGE (q)-[:USES_TABLE]->(t)
            """, {'qid': item['id'], 'tname': table_name})

    print(f"✅ Successfully processed and created/updated {len(all_questions)} SampleQuestion nodes.")


def build_graph():
    """Main function to build the entire knowledge graph in FalkorDB."""
    schema_data = get_schema_data()

    # connect to FalkorDB
    falkor_db = FalkorDB(host='localhost', port=6379)
    graph = falkor_db.select_graph(config.GRAPH_NAME)

    text_embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

    print(f"--- Clearing existing graph '{config.GRAPH_NAME}' ---")
    try:
        graph.delete()
        # Re-select the graph after deleting
        graph = falkor_db.select_graph(config.GRAPH_NAME)
    except Exception as e:
        print(f"Graph did not exist, creating new. (Error: {e})")

    _upload_schema_to_falkordb(graph, schema_data)
    _create_sample_question_nodes(graph, text_embedding_model)
    _run_han_pipeline_falkordb(graph, text_embedding_model)

    print("\n--- Verification ---")
    result = graph.query("MATCH (t:Table) RETURN t.name, size(t.han_embedding) AS embedding_size").result_set
    if result:
        print("Successfully verified embeddings in FalkorDB:")
        for record in result:
            print(f"  - Table: {record[0]}, Embedding Size: {record[1]}")
    else:
        print("Verification failed. No embeddings found.")


if __name__ == "__main__":
    build_graph()
