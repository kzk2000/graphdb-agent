import re

import litellm

from graphdb_agent import config, retriever, prompt_engine

litellm.suppress_debug_info = True  # turns of prints of "Provider List: https://docs.litellm.ai/docs/providers"
# import os; os.environ['LITELLM_LOG'] = 'DEBUG'

def format_sql_query(sql: str, indent: int = 2) -> str:
    # Define clause keywords that should start on a new line
    clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT',
               'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN']

    # Sort by length to match longer keywords like 'GROUP BY' before 'GROUP'
    clauses_sorted = sorted(clauses, key=len, reverse=True)

    # Normalize whitespace
    sql_clean = re.sub(r'\s+', ' ', sql.strip())

    # Inject a newline before each clause (and preserve any trailing content)
    for clause in clauses_sorted:
        pattern = re.compile(rf'\b({re.escape(clause)})\b', re.IGNORECASE)
        sql_clean = pattern.sub(r'\n\1', sql_clean)

    # Clean up leading/trailing space and split into lines
    lines = [line.strip() for line in sql_clean.strip().split('\n') if line.strip()]

    formatted_lines = []
    for line in lines:
        upper_line = line.upper()
        is_clause = any(upper_line.startswith(c) for c in clauses_sorted)
        if is_clause:
            # Split keyword and the rest of the line
            for clause in clauses_sorted:
                if upper_line.startswith(clause):
                    keyword_len = len(clause)
                    formatted_lines.append(clause)
                    rest = line[keyword_len:].strip()
                    if rest:
                        formatted_lines.append(' ' * indent + rest)
                    break
        else:
            formatted_lines.append(' ' * indent + line)

    return '\n'.join(formatted_lines)



def get_sql_from_llm(prompt: str) -> str:
    """
    Sends the generated prompt to the Google Gemini model using litellm
    and returns the generated SQL query.
    """
    try:
        # litellm uses a unified 'completion' function for all providers.
        # You specify the model using the 'model' parameter.
        # For Google, the model name is prefixed with "gemini/".
        response = litellm.completion(
            model=config.LLM_MODEL,  # Use a fast and capable model
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Set to 0.0 for deterministic, factual SQL generation
            max_tokens=500,  # Max length of the generated SQL query
            stream=False,  # We want the full response at once
            api_key=config.API_KEY,
        )

        # The response object is a dictionary-like object.
        # The actual text content is in choices[0].message.content
        raw_sql = response.choices[0].message.content

        # Clean up the response: remove markdown code blocks if the LLM adds them
        if raw_sql.strip().startswith("```sql"):
            raw_sql = raw_sql.strip()[5:]  # Remove ```sql
            if raw_sql.endswith("```"):
                raw_sql = raw_sql[:-3]  # Remove ```

        return raw_sql.strip()

        # formatted_sql = format_sql_query(raw_sql.strip())
        #return formatted_sql


    except Exception as e:
        # Handle potential API errors (e.g., invalid key, network issues)
        print(f"\n❌ An error occurred while calling the LLM API: {e}")
        return f"-- ERROR: Could not generate SQL. Reason: {e}"


def run_query_pipeline(user_query: str):
    """
    Orchestrates the full text-to-SQL pipeline.
    """
    # 1. Retrieve candidate tables from the graph
    # For a complex query, you might want to retrieve more candidates
    candidate_tables = retriever.find_candidate_tables(user_query, limit=4)

    if not candidate_tables:
        print("\nCould not identify any relevant tables for the query. Aborting.")
        return

    # 2. Generate the prompt using the candidates
    print("\n⚙️  Generating LLM prompt...")
    prompt = prompt_engine.generate_llm_prompt(user_query, candidate_tables)

    print("\n" + "=" * 80)
    print("FINAL LLM PROMPT:")
    print("=" * 80)
    print(f"User query: {prompt}")
    print("=" * 80)

    # 3. Send prompt to LLM and get SQL (This is now a real call)
    print("\n🤖 Sending prompt to Gemini for SQL generation...")
    sql_query = get_sql_from_llm(prompt)

    print("\n" + "=" * 80)
    print(user_query)
    print(f"✅ Generated {config.SQL_DIALECT} Query:")
    print("=" * 80)
    print(sql_query)
    print("=" * 80)

    # 4. (Conceptual) Execute SQL and return result as pd.Dataframe
    # ...


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run the Graph Text-to-SQL Agent.")
    # parser.add_argument("query", type=str, help="The user's natural language query.")
    # args = parser.parse_args()

    user_query = 'Show me the names of customers who placed 3 largest orders in April 2025?'
    user_query = "Which 2 products have the largest month to month growth since April 2024"
    #user_query = "Tell me the top 3 most revenue spike days"
    #user_query = "Who are the largest suppliers?"
    run_query_pipeline(user_query)
