# Graph-Powered Text-to-SQL Agent

An advanced Text-to-SQL agent that uses a Knowledge Graph (FalkorDB) and a Heterogeneous Graph Attention Network (HAN) to accurately identify relevant tables before generating SQL with a Large Language Model (LLM).

This project implements a two-stage Retrieval-Augmented Generation (RAG) architecture for a more accurate and efficient Text-to-SQL conversion.

---

## Key Features

-   **YAML-based Schema:** Easily define and manage your database schema, descriptions, and relationships using simple YAML files.
-   **Heterogeneous Knowledge Graph:** Models your SQL schema as a rich graph in FalkorDB with different node (`Table`, `Column`) and edge (`HAS_COLUMN`, `LINKS_TO`) types.
-   **State-of-the-Art Embeddings:** Uses a Heterogeneous Graph Attention Network (HAN) to create powerful, context-aware vector embeddings for each table. These embeddings understand both the table's content and its relationships within the schema.
-   **Two-Stage RAG Architecture:**
    1.  **Retrieval:** A fast vector search in FalkorDB finds the most relevant candidate tables for a user query.
    2.  **Generation:** A precisely structured prompt containing only the candidate table schemas is sent to an LLM to generate the final SQL query.

---

## Architecture Overview

The system is split into two main phases: an offline **Indexing Pipeline** and an online **Query Pipeline**.

### 1. Indexing Pipeline (Offline)

This is a one-time process run by `graph_builder.py` to prepare the knowledge graph.

  <!-- You can create a simple diagram for this -->

1.  **Parse Schemas:** Reads all `.yaml` files from the `/schemas` directory.
2.  **Build Graph:** Populates a FalkorDB database with `Table` and `Column` nodes and their relationships.
3.  **Generate Embeddings:** Runs the HAN model on the graph to compute a `han_embedding` vector for each `Table` node and stores it back in FalkorDB.

### 2. Query Pipeline (Online)

This is the real-time process run by `agent.py` to answer a user's question.

 <!-- You can create a simple diagram for this -->

1.  **User Query:** The user provides a question in natural language (e.g., "who were our top 5 customers last month?").
2.  **Retrieve Candidates:** The `retriever.py` module embeds the user query and performs a cosine similarity search in FalkorDB to find the most relevant `Table` nodes based on their `han_embedding`.
3.  **Generate Prompt:** The `prompt_engine.py` module takes the list of candidate tables, loads their full schemas from the YAML files, and constructs a detailed, structured prompt for the LLM.
4.  **Generate SQL:** The prompt is sent to an LLM (e.g., GPT-4, Claude 3), which returns the final, executable SQL query.

---

## Project Structure

```
.
├── docker-compose.yml
├── docs                                                    -- folder: contains documentation on some of the design desciions
├── LICENSE
├── pyproject.toml
├── README.md
├── schemas                                                 -- folder: contains SQL table definitions + meta data  
├── src
│   ├── graphdb_agent
│   │   ├── agent.py
│   │   ├── config.py
│   │   ├── generate_starfish_questions.py
│   │   ├── graph_builder.py
│   │   ├── __init__.py
│   │   ├── lemmatization.py
│   │   ├── prompt_engine.py
│   │   ├── retriever.py
│   │   ├── semantic.py
│   │   └── z_snippets.py
└── uv.lock                                                 -- for `uv sync`

```

---

## Setup and Installation
**Installation Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/graphdb-agent.git
    cd graphdb-agent
    ```

2.  **Create and activate a virtual environment via uv (assumes Linux or Mac):**
    ```bash
    uv sync                     # creates folder .venv with everything you need
    source .venv/bin/activate   
    ```
3.  **Configure the application:**<br>
    Open `graphdb-agent/config.py` and update your settings, in particular: 
    - Choice of `SQL_DIALECT` 
    - LiteLLM `API_KEY` 
    - LiteLLM `LLM_MODEL`

4. **Start the FalkorDB server:**
   ```bash
   docker compose up -d
   ``` 
   Its UI is accessible via http://localhost:3000/graph 
---

## Usage

### 1. Define Your Schema

Add or edit the `.yaml` files in the `/schemas` directory to match your database schema. Ensure you provide clear 
descriptions and define `_global_join_paths.yaml` for foreign key relationships, too. Follow the layout of the 
existing examples.

### 2. Run the Indexing Pipeline

Execute the graph builder script to populate FalkorDB and generate the HAN embeddings. This only needs to be done once, or whenever your schema changes.

```bash
uv run src/graphdb_agent/graph_builder.py 
```

### 3. Run a Query

Execute the main agent script to process a natural language query.

```bash
# The agent.py script can be modified to take a command-line argument
uv run src/graphdb_agent/agent.py "Show me the names of customers who placed large orders"
```
This will output the final, ready-to-use SQL query.

