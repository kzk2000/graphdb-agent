import os
import yaml
from typing import List, Dict, Any
from graphdb_agent import config, retriever


# This helper function remains unchanged
def _generate_create_table_statement(table_name: str, table_data: Dict[str, Any]) -> str:
    parts = [f"-- Table: {table_name}"]
    if table_data.get('table_description'):
        parts.append(f"-- Description: {table_data['table_description']}")

    parts.append(f"CREATE TABLE {table_name} (")
    column_definitions = []
    for col in table_data.get('columns', []):
        col_name = col.get('name', 'unknown_column')
        col_type = col.get('type', 'TEXT').upper()
        col_desc = col.get('description', '')
        synonyms = col.get('synonyms', [])  # <-- Get synonyms

        is_primary_key = "primary key" in col_desc.lower()

        definition = f"    {col_name} {col_type}{' PRIMARY KEY' if is_primary_key else ''}"

        # Build the comment string
        comment_parts = [col_desc]
        if synonyms:
            comment_parts.append(f"Synonyms: {', '.join(synonyms)}")

        full_comment = " ".join(filter(None, comment_parts))  # Join non-empty parts

        if full_comment:
            definition += f", -- {full_comment}"

        column_definitions.append(definition)
    parts.append(",\n".join(column_definitions))
    parts.append(");")
    return "\n".join(parts)


def generate_llm_prompt(user_query: str, candidate_tables: List[str]) -> str:
    """
    Generates a complete, structured LLM prompt for the text-to-SQL task,
    using a global join configuration file.
    """

    # --- 1. Load Schemas for Candidate Tables ---
    schema_statements = []
    for table_name in candidate_tables:
        yaml_path = os.path.join(config.SCHEMA_DIR, f"{table_name}.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                table_data = yaml.safe_load(f)
                schema_statements.append(_generate_create_table_statement(table_name, table_data))
        else:
            print(f"Warning: YAML file not found for table '{table_name}'.")

    full_schema_str = "\n\n".join(schema_statements)

    # --- 2. Load and Filter Global Join Information ---
    join_statements = []
    join_config_path = os.path.join(config.SCHEMA_DIR, "join_paths.yaml")
    candidate_set = set(candidate_tables)

    if os.path.exists(join_config_path):
        with open(join_config_path, 'r') as f:
            all_joins = yaml.safe_load(f) or []  # Handle empty file
            for join in all_joins:
                # Only include the join path if BOTH tables are in the candidate list
                if join['from_table'] in candidate_set and join['to_table'] in candidate_set:
                    join_str = (f"-- To join `{join['from_table']}` and `{join['to_table']}`, use: "
                                f"`{join['from_table']}.{join['from_column']}` = `{join['to_table']}.{join['to_column']}`")
                    join_statements.append(join_str)

    # --- 3. Assemble the Full Prompt ---
    join_info_section = ""
    if join_statements:
        unique_joins = sorted(list(set(join_statements)))
        join_info_section = f"""
### JOIN INFORMATION
Use the following statements to join the tables:
{chr(10).join(unique_joins)}
"""

    # The prompt template itself remains the same
    prompt = f"""You are an expert {config.SQL_DIALECT} data analyst. Your sole purpose is to write correct, efficient, and executable SQL queries based on a provided schema and a user's question.

### TASK
Convert the user's question into a single, executable {config.SQL_DIALECT} query.

### DATABASE SCHEMA
You must only use the following tables and columns.

{full_schema_str}
{join_info_section}
### USER QUESTION
"{user_query}"

### INSTRUCTIONS
1.  **Join Correctly:** If multiple tables are needed, use the explicit join paths provided in the "JOIN INFORMATION" section.
2.  **Interpret Ambiguity:** If the user's query is subjective (e.g., "large", "recent", "top"), make a reasonable assumption. For example, interpret "large orders" as orders in the top 10% by value or those over a fixed threshold like 1000.
3.  **Select Correct Columns:** Ensure the columns in the `SELECT` statement directly answer the user's question.
4.  **Output Format:** Respond with ONLY the raw SQL query. Do not include any explanations, comments outside of the SQL, or markdown formatting like ```sql.
"""

    return prompt


# --- Main Execution Block for Demonstration ---
if __name__ == "__main__":
    user_query = "Show me the names of customers who placed large orders"
    han_selected_tables = ["customers", "orders"]
    #han_selected_tables = find_candidate_tables(user_question, 3)
    han_selected_tables = retriever.find_candidate_tables_hybrid(user_query, 3)

    print(f"--- Generating LLM Prompt for query: '{user_query}' ---")
    print(f"--- Using candidate tables: {han_selected_tables} ---\n")

    final_prompt = generate_llm_prompt(user_query, han_selected_tables)

    print("=" * 80)
    print("REFINED LLM PROMPT (using global joins):")
    print("=" * 80)
    print(final_prompt)
    print("=" * 80)
