import litellm

from graphdb_agent import config, semantic

litellm.suppress_debug_info = True


def generate_starfish_questions(table_schema: str) -> list[str]:
    """
    Uses an LLM to generate a diverse set of sample questions for a given table schema.
    """

    prompt_template = """You are an expert Business Intelligence (BI) analyst and prompt engineer. Your task is to generate a diverse set of high-quality sample questions for a given SQL table schema.

These questions will be used to train a text-to-SQL retrieval system, so their diversity is critical for teaching the system about the full range of the table's capabilities.

### TASK
Generate exactly 7 sample questions based on the provided table schema.

### INPUT TABLE SCHEMA
{table_schema}

### CORE PRINCIPLES FOR DIVERSITY ("The Semantic Starfish")
Your set of 7 questions must be diverse and cover different analytical patterns. Do not just ask the same type of question in different ways. Ensure your questions cover a mix of the following patterns:

1.  **Simple Filtering/Lookup:** Questions that filter on a single column's value (e.g., "List all products in the 'Electronics' category").
2.  **Simple Aggregation:** Questions that use a single aggregate function like COUNT, SUM, or AVG across the whole table (e.g., "What is the average price of all products?").
3.  **Grouping & Aggregation:** Questions that group by a categorical column and then perform an aggregation (e.g., "What is the average price per category?").
4.  **Temporal Analysis:** If there is a date or timestamp column, questions about trends, specific time periods, or recency (e.g., "How many orders were placed last month?").
5.  **Ranking & Ordering:** Questions that ask for top/bottom N results (e.g., "What are our 5 most expensive products?").
6.  **Complex Combinations:** Questions that combine multiple principles, such as filtering, grouping, and aggregation (e.g., "What was the total revenue from 'Laptops' in the last quarter?").
7.  **Implied Joins:** If a column is clearly a foreign key (e.g., `customer_id`, `supplier_id`), formulate a question that would require that join (e.g., "Which supplier provides the most products?").

### CRITICAL INSTRUCTION
**DO NOT** generate simple rephrasings of the same question. Each of the 7 questions should test a different analytical capability of the table.

### OUTPUT FORMAT
Respond with ONLY a numbered list of the 7 questions. Do not add any explanations, introductions, or other text.
"""

    prompt = prompt_template.format(table_schema=table_schema)

    try:
        response = litellm.completion(
            model=config.LLM_MODEL, #"gemini/gemini-1.5-pro-latest",  # Use a powerful model for this creative task
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,  # Allow for some creativity
            api_key=config.API_KEY,
            stream=False,
        )

        content = response.choices[0].message.content
        # Split the numbered list into individual questions
        questions = [line.split('.', 1)[1].strip() for line in content.strip().split('\n') if line.strip()]
        return questions

    except Exception as e:
        print(f"Error generating questions: {e}")
        return []


# --- Example Usage ---
if __name__ == '__main__':
    # You would generate this schema string from your YAML files
    table_schema = """
-- Table: orders
-- Description: Records every sales transaction made by customers, including financial totals and dates.
CREATE TABLE orders (
    id INT PRIMARY KEY, -- Unique identifier for an order, primary key.
    customer_id INT, -- Foreign key that links to the customers table id.
    order_date DATE, -- The date the transaction occurred.
    total_amount DECIMAL -- The total monetary value of the order.
);
"""
    print("--- Generating Starfish Questions for the 'orders' table ---")
    generated_questions = generate_starfish_questions(table_schema)

    if generated_questions:
        print("\nRecommended Diverse Questions:")
        for i, q in enumerate(generated_questions):
            print(f"{i + 1}. {q}")

    semantic.visualize_question_diversity( generated_questions[0], generated_questions, title='Orders')
