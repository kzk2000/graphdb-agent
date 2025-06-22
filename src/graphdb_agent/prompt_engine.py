import os
import re
from typing import List, Dict, Any

import yaml

from graphdb_agent import config, retriever


# In text_to_sql_agent/prompt_engine.py

# In text_to_sql_agent/prompt_engine.py

def _generate_create_table_statement(table_name: str, table_data: Dict[str, Any]) -> str:
    """
    Generates the state-of-the-art, fully aligned and hierarchically structured
    CREATE TABLE statement for the LLM prompt.
    """
    # --- Setup ---
    parts = [f"-- Table: {table_name}"]
    description = table_data.get('table_description')
    if description:
        parts.append("-- Description:")
        if isinstance(description, list):
            for line in description:
                parts.append(f"--   - {line}")
        else:
            parts.append(f"--   - {description}")
    parts.append(f"CREATE TABLE {table_name} (")

    # --- First Pass: Calculate layout ---
    column_data_for_formatting = []
    max_len = 0
    for col in table_data.get('columns', []):
        base_definition = f"    {col.get('name', '')} {col.get('type', 'TEXT').upper()}{' PRIMARY KEY' if 'primary key' in col.get('description', '').lower() else ''}"
        if len(base_definition) > max_len:
            max_len = len(base_definition)

        # Structure comments as a list of blocks ---
        comment_blocks = []
        if col.get('description'):
            comment_blocks.append([col['description'].strip().rstrip('.')])
        if col.get('synonyms'):
            comment_blocks.append([f"Synonyms: {', '.join(col['synonyms'])}"])
        if col.get('rules'):
            rule_block = ["Rules:"]
            for rule in col['rules']:
                rule_block.append(f"  - {rule}")
            comment_blocks.append(rule_block)

        column_data_for_formatting.append({
            "base": base_definition,
            "comment_blocks": comment_blocks
        })

    # --- Second Pass: Build the formatted strings ---
    column_definitions = []
    for data in column_data_for_formatting:
        base_def = data['base']
        comment_blocks = data['comment_blocks']

        final_column_str = base_def
        if comment_blocks:
            # Add the first line of the first block on the same line
            comma_or_space = " " if data == column_data_for_formatting[-1] else ","
            final_column_str += f"{comma_or_space}{' ' * (max_len - len(base_def))} -- {comment_blocks[0][0]}"

            # Calculate padding for all subsequent comment lines
            comment_padding = ' ' * (max_len + len(", -- "))

            # Add the rest of the first block
            for part in comment_blocks[0][1:]:
                final_column_str += f"\n{comment_padding}{part}"

            # Add all subsequent blocks, prefixed with the list marker '-'
            for block in comment_blocks[1:]:
                for i, part in enumerate(block):
                    # The first line of the new block gets the list marker
                    prefix = "- " if i == 0 else "  "
                    final_column_str += f"\n{comment_padding}{prefix}{part}"

        column_definitions.append(final_column_str)

    parts.append(",\n".join(column_definitions))
    parts.append(");")

    return "\n".join(parts)


def generate_join_info_section(candidate_tables: list, all_joins: list) -> str:
    """
    Generates the formatted 'JOIN INFORMATION' section for the LLM prompt
    by filtering global joins against the candidate tables.

    Returns the formatted string section or an empty string if no joins are relevant.
    """
    join_statements = []
    candidate_set = set(candidate_tables)  # Use a set for efficient O(1) lookups

    for join in all_joins:
        # The core logic: Only include the join if BOTH tables are in the candidate list.
        if join.get('from_table') in candidate_set and join.get('to_table') in candidate_set:
            # Format the join information into a clear, human-readable comment
            join_str = (
                f"-- To join `{join['from_table']}` and `{join['to_table']}`, use: "
                f"`{join['from_table']}.{join['from_column']}` = `{join['to_table']}.{join['to_column']}`"
            )
            join_statements.append(join_str)

    # If no relevant joins were found, return an empty string to keep the prompt clean.
    if not join_statements:
        return ""

    # Create the final, formatted section with a clear header.
    # Using a set to remove potential duplicates and sorting for deterministic output.
    unique_joins = sorted(list(set(join_statements)))

    # The chr(10) is a newline character, making the output clean.
    join_info_block = "\n".join(unique_joins)

    return f"""
### JOIN INFORMATION
Use the following statements to join the tables:
{join_info_block}
"""


def generate_llm_prompt(user_query: str, candidate_tables: List[str]) -> str:
    """
    Generates a complete, structured LLM prompt for the text-to-SQL task,
    using a global join configuration file.
    """

    # --- 1. Load Schema and All Potential Business Rules ---
    schema_statements = []
    all_business_rules = []
    candidate_data = {}  # Store the full data for the sanity check

    for table_name in candidate_tables:
        yaml_path = os.path.join(config.SCHEMA_DIR, f"{table_name}.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                table_data = yaml.safe_load(f)
                candidate_data[table_name] = table_data

                schema_statements.append(_generate_create_table_statement(table_name, table_data))

                # Collect all rules from the selected tables
                if table_data.get('business_rules'):
                    all_business_rules.extend(table_data['business_rules'])

    full_schema_str = "\n\n".join(schema_statements)

    # --- 2. Load and Filter Global Join Information ---
    join_statements = []
    join_config_path = os.path.join(config.SCHEMA_DIR, "_global_join_paths.yaml")
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

    # --- 3. Business rules ---
    business_rules_statements = []
    for table_name in candidate_tables:
        yaml_path = os.path.join(config.SCHEMA_DIR, f"{table_name}.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                table_data = yaml.safe_load(f)
                rules = table_data.get('business_rules', [])
                if rules:
                    business_rules_statements.extend([f"- {rule}" for rule in rules])

    business_rules_section = ""
    if business_rules_statements:
        rules_block = "\n".join(business_rules_statements)
        business_rules_section = f"""
### BUSINESS RULES & DEFINITIONS
You must follow these rules to correctly interpret business terms:
{rules_block}
"""

    # --- 4. Assemble the Full Prompt ---
    join_info_section = ""
    if join_statements:
        unique_joins = sorted(list(set(join_statements)))
        join_info_section = f"""
### JOIN INFORMATION
Use the following statements to join the tables:
{chr(10).join(unique_joins)}   
"""
    # --- 5. Load and assemble Few-Shot Examples ---
    few_shot_examples = []
    for table_name in candidate_tables:
        yaml_path = os.path.join(config.SCHEMA_DIR, f"{table_name}.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                table_data = yaml.safe_load(f)
                examples = table_data.get('few_shot_examples', [])
                if examples:
                    few_shot_examples.extend(examples)

    examples_section = ""
    if few_shot_examples:
        # De-duplicate examples in case multiple tables provide the same one
        unique_examples_str = {f"Question: {ex['question']}\nSQL:\n{ex['sql']}" for ex in few_shot_examples}

        examples_block = "\n---\n".join(unique_examples_str)
        examples_section = f"""
### EXAMPLES
Here are some examples of how to translate questions into {config.SQL_DIALECT} queries for this schema. Follow these patterns closely.

{examples_block}
"""

    # The prompt template itself remains the same
    prompt = f"""You are an expert {config.SQL_DIALECT} data analyst. Your sole purpose is to write correct, efficient, and executable SQL queries based on a provided schema and a user's question.

### TASK
Convert the user's question into a single, executable {config.SQL_DIALECT} query.

### DATABASE SCHEMA
You must only use the following tables and columns.

{full_schema_str}
{join_info_section}
{business_rules_section}
{examples_section}
### USER QUESTION
"{user_query}"

### INSTRUCTIONS
1.  **Join Correctly:** If multiple tables are needed, use the explicit join paths provided in the "JOIN INFORMATION" section.
2.  **Interpret Ambiguity:** If the user's query is subjective (e.g., "large", "recent", "top"), make a reasonable assumption. For example, interpret "large orders" as orders in the top 10% by value or those over a fixed threshold like 1000.
3.  **Select Correct Columns:** Ensure the columns in the `SELECT` statement directly answer the user's question.
4.  **Output Format:** Respond with ONLY the raw SQL query. Do not include any explanations, comments outside of the SQL, or markdown formatting like ```sql.
5.  **Format the SQL:** Format the final SQL query for readability. Use uppercase for all SQL keywords (e.g., `SELECT`, `FROM`, `WHERE`). Indent subqueries and clauses with 2 whitespaces logically.
"""

    return prompt


# --- Main Execution Block for Demonstration ---
if __name__ == "__main__":
    user_query = "Show me the names of customers who placed large orders"

    user_query = "Who are the largest suppliers?"

    #    candidate_tables = ["customers"]

    han_selected_tables = ["customers", "orders", "suppliers"]
    # han_selected_tables = find_candidate_tables(user_question, 3)
    #han_selected_tables = retriever.find_candidate_tables_hybrid(user_query, 3)

    print(f"--- Generating LLM Prompt for query: '{user_query}' ---")
    print(f"--- Using candidate tables: {han_selected_tables} ---\n")

    final_prompt = generate_llm_prompt(user_query, han_selected_tables)

    print("=" * 80)
    print("LLM PROMPT:")
    print("=" * 80)
    print(final_prompt)
    print("=" * 80)
