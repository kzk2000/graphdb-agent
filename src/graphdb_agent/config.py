import os

# --- FalkorDB Configuration ---
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", 6379))
GRAPH_NAME = "text2sql_graph"

# --- Neo4j Configuration ---
# It's best practice to load sensitive data from environment variables in production
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_secret_password") # IMPORTANT: Change this or set env variable

# --- Model Configuration ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
HAN_HIDDEN_CHANNELS = 64
HAN_OUT_CHANNELS = EMBEDDING_DIM # ensure this remains equal to the input

# --- File Paths ---
# Assumes the script is run from the root of the project
SCHEMA_DIR = "/home/user/Dropbox/Python/PycharmProjects/graphdb-agent/schemas"

# --- LLM Configuration ---
# SQL_DIALECT = "PostgreSQL"
SQL_DIALECT = "Snowflake"

# --- LLMLITE params (used in agent.py)
LLM_MODEL = "gemini/gemini-1.5-flash-latest"
API_KEY=os.environ["GOOGLE_API_KEY"]