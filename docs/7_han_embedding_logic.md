Excellent question. You've correctly identified that as your YAML files become richer with metadata, your embedding strategy must evolve to incorporate all of it. If you don't, your retrieval system won't benefit from the valuable context you've added.

The goal is to create a single, comprehensive "text document" for each `Table` and `Column` node before you embed it. This document should be a string concatenation of all the relevant metadata.

Let's walk through the necessary changes to your `graph_builder.py` script.

---

### The Strategy: Create a "Full Context Document" for Embedding

We will modify the `_run_han_pipeline_falkordb` (or Neo4j) function to build a much richer text string for both tables and columns.

#### 1. For `Table` Nodes:

The text document for a `Table` node should now include:
*   Its name.
*   Its multi-line description.
*   Its table-level business rules.
*   The *questions* from its few-shot examples (but not the SQL).

**Why not the SQL?** The SQL code is procedural and contains keywords (`SELECT`, `FROM`) that can add noise to the semantic embedding. The natural language *question* is the most valuable part for matching user intent.

#### 2. For `Column` Nodes:

The text document for a `Column` node should now include:
*   Its name.
*   Its description.
*   Its data type.
*   Its synonyms.
*   Its column-level rules.

---

### The Refactored `graph_builder.py`

Here is the key part of the `_run_han_pipeline...` function, updated to build these rich documents.

```python
# In text_to_sql_agent/graph_builder.py

# ... (imports and HANModel class are the same) ...

def _run_han_pipeline_falkordb(graph, text_embed_model):
    print("--- Starting HAN Embedding Pipeline ---")
    
    # Step 1: Extract graph data from FalkorDB
    # We need to fetch all the new metadata properties we've added.
    print("Extracting graph data with all metadata...")
    tables_res = graph.query("""
        MATCH (t:Table) 
        RETURN id(t) AS id, t.name AS name, t.description AS desc, 
               t.sample_questions AS questions, t.business_rules AS rules, 
               t.few_shot_examples AS examples
    """).result_set
    
    cols_res = graph.query("""
        MATCH (c:Column) 
        RETURN id(c) AS id, c.name AS name, c.description AS desc, 
               c.type AS type, c.synonyms AS synonyms, c.rules AS rules
    """).result_set
    
    # ... (queries for has_col_res and links_to_res remain the same) ...
    has_col_res = ...
    links_to_res = ...

    # Map results to dictionaries for easier access
    tables_data = [{'id': r[0], 'name': r[1], 'desc': r[2], 'questions': r[3], 'rules': r[4], 'examples': r[5]} for r in tables_res]
    cols_data = [{'id': r[0], 'name': r[1], 'desc': r[2], 'type': r[3], 'synonyms': r[4], 'rules': r[5]} for r in cols_res]
    # ... (map has_col_data and links_to_data) ...
    
    table_id_map = {r['id']: i for i, r in enumerate(tables_data)}
    col_id_map = {r['id']: i for i, r in enumerate(cols_data)}

    # --- Step 2: Prepare HeteroData object with RICH text features ---
    print("Preparing rich text documents for embedding...")
    data = HeteroData()
    
    # --- Build Rich Table Documents ---
    table_texts = []
    for r in tables_data:
        parts = []
        parts.append(f"Table name: {r['name']}.")
        
        # Add multi-line description
        if r.get('desc') and isinstance(r['desc'], list):
            parts.append(f"Description: {' '.join(r['desc'])}")
        elif r.get('desc'):
            parts.append(f"Description: {r['desc']}")
            
        # Add business rules
        if r.get('rules'):
            parts.append(f"Business Rules: {' '.join(r['rules'])}")
            
        # Add sample questions
        if r.get('questions'):
            parts.append(f"This table can answer questions like: '{' '.join(r['questions'])}'")
            
        # Add questions from few-shot examples
        if r.get('examples'):
            example_questions = [ex['question'] for ex in r['examples']]
            parts.append(f"It is also used for complex questions such as: '{' '.join(example_questions)}'")
            
        table_texts.append(" ".join(parts))

    # --- Build Rich Column Documents ---
    col_texts = []
    for r in cols_data:
        parts = []
        parts.append(f"Column name: {r['name']}.")
        parts.append(f"Data type: {r.get('type', 'TEXT')}.")
        
        if r.get('desc'):
            parts.append(f"Description: {r['desc']}")
        
        if r.get('synonyms'):
            parts.append(f"Synonyms: {', '.join(r['synonyms'])}.")
            
        if r.get('rules'):
            parts.append(f"Rules: {' '.join(r['rules'])}")
            
        col_texts.append(" ".join(parts))
        
    # --- Embed the rich documents ---
    print("Embedding rich documents...")
    data['table'].x = torch.tensor(text_embed_model.encode(table_texts), dtype=torch.float)
    data['column'].x = torch.tensor(text_embed_model.encode(col_texts), dtype=torch.float)
    
    # ... (The rest of the pipeline for creating edge_index, running the HAN model,
    #      and writing embeddings back to the database remains exactly the same) ...
```

### Important Prerequisite: Update `_upload_schema_to_...`

This new embedding logic assumes that all the rich metadata from your YAML files has already been saved as properties on the nodes in your graph database. You must first ensure your `_upload_schema_to_falkordb` (or Neo4j) function is saving everything.

**In `graph_builder.py`, inside `_upload_schema_to_falkordb`:**

```python
# ... (inside the loop over tables) ...
for table_name, data in schema_data.items():
    # Update the Table node creation to include all new keys
    graph.query("""
        MERGE (t:Table {name: $name})
        SET t.description = $desc, 
            t.sample_questions = $questions,
            t.business_rules = $rules,
            t.few_shot_examples = $examples
    """, {
        'name': table_name, 
        'desc': data.get('table_description', []),
        'questions': data.get('sample_questions', []),
        'rules': data.get('business_rules', []),
        'examples': data.get('few_shot_examples', [])
    })
    
    # Update the Column node creation
    for col in data['columns']:
        graph.query("""
            MATCH (t:Table {name: $table_name})
            MERGE (c:Column {name: $col_name, table_name: $table_name})
            SET c.description = $desc, 
                c.type = $type, 
                c.synonyms = $synonyms,
                c.rules = $rules
            # ... (MERGE relationships) ...
        """, {
            'table_name': table_name, 
            'col_name': col['name'], 
            'desc': col.get('description', ''), 
            'type': col.get('type', 'TEXT'),
            'synonyms': col.get('synonyms', []),
            'rules': col.get('rules', [])
        })
```

### The Result

By making these changes, you ensure that every piece of valuable, human-curated metadata is "baked into" the initial vector embeddings.

*   The `customers` table embedding will now be semantically aware of concepts like "active user," "new customer," and "engagement score" because those terms are now part of the text document that gets embedded.
*   The `total_amount` column embedding will be aware of "revenue" and "sales."
*   The `status` column embedding will be aware of the business logic for "Shipped" and "Delivered."

This creates a much more intelligent "semantic surface" for your retriever to search against, dramatically improving its ability to find the correct tables even for complex or abstract user queries.
