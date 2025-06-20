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
        graph.query("""
            MERGE (t:Table {name: $name})
            SET t.description = $desc, t.sample_questions = $questions
        """, {'name': table_name, 'desc': data['table_description'], 'questions': data['sample_questions']})

        # Create Column nodes and link them to their parent table
        for col in data['columns']:
            graph.query("""
                 MATCH (t:Table {name: $table_name})
                 MERGE (c:Column {name: $col_name, table_name: $table_name})
                 MERGE (t)-[:HAS_COLUMN]->(c)
                 MERGE (c)-[:BELONGS_TO]->(t)                 
                 SET c.description = $col_desc, c.type = $col_type, c.synonyms = $synonyms
             """, {
                'table_name': table_name,
                'col_name': col['name'],
                'col_desc': col.get('description', ''),
                'col_type': col.get('type', 'TEXT'),
                'synonyms': ",".join(col.get('synonyms', []))
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
    tables_res = graph.query(
        "MATCH (t:Table) RETURN id(t) AS id, t.name AS name, t.description AS desc, t.sample_questions AS questions").result_set
    tables_data = [{'id': r[0], 'name': r[1], 'desc': r[2], 'sample_questions': r[3]} for r in tables_res]

    cols_res = graph.query(
        "MATCH (c:Column) RETURN id(c) AS id, c.name AS name, c.description AS desc, c.type AS type").result_set
    cols_data = [{'id': r[0], 'name': r[1], 'desc': r[2], 'type': r[3]} for r in cols_res]

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

    # Step 2: Prepare HeteroData object for PyG
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

    table_texts = []
    for r in tables_data:
        questions_str = ""
        if r.get('sample_questions'):
            questions_str = f"This table can answer questions like: '{' '.join(r['sample_questions'])}'"

        text = (f"Table: {r['name']}; Description: {r['desc']} {questions_str}")
        table_texts.append(text)

    data['table'].x = torch.tensor(text_embedding_model.encode(table_texts), dtype=torch.float)

    col_texts = []
    for r in cols_data:
        synonyms_str = ""
        if r.get('synonyms'):
            synonyms_str = f"Also known as: {', '.join(r['synonyms'])}."

        text = (f"Column: {r['name']}. Description: {r['desc']} "
                f"Type: {r['type']}. {synonyms_str}")
        col_texts.append(text)

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
                if "join_paths" in filename:
                    schema_data["join_paths"] = data
                else:
                    schema_data['tables'][data['table_name']] = data
    return schema_data


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
