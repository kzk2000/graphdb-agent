import argparse
from graphdb_agent import retriever, prompt_engine

if False:
    # used for local testing
    user_query = "What customers had the most orders?"

# In a real application, you would import your LLM client here
# from . import llm_client

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
    print(prompt)
    print("=" * 80)

    # 3. (Conceptual) Send prompt to LLM and get SQL
    print("\n🤖 Sending prompt to LLM for SQL generation...")
    # sql_query = llm_client.get_sql(prompt)
    # print(f"\nGenerated SQL:\n{sql_query}")
    print("\n(LLM call is conceptual in this example.)")

    # 4. (Conceptual) Execute SQL and return result
    # ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Graph Text-to-SQL Agent.")
    parser.add_argument("query", type=str, help="The user's natural language query.")
    args = parser.parse_args()

    run_query_pipeline(args.query)
