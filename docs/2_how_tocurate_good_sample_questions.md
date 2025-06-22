**Question:**<br>
Regarding "Aim for 3 to 7 high-quality, diverse sample questions per table.  Avoid adding a dozen slight variations of the same question."
is there a way to create sample questions that have maximum diversity in embedding space while representing meaningful semantics?

**Answer:**<br>
That is a brilliant, next-level question. You've hit on a core challenge in machine learning and data augmentation: how to generate a small, high-quality, and diverse dataset that provides maximum signal with minimum noise.

Yes, there is absolutely a way to do this programmatically. The best approach is a multi-stage pipeline that combines the generative power of LLMs with the analytical precision of vector mathematics.

Here is a sophisticated, step-by-step workflow to create a set of sample questions that have maximum diversity in embedding space while remaining semantically meaningful.

---

### The Goal: The "Semantic Starfish"

Imagine the core meaning of your `orders` table is a point in space. You don't want a tight cluster of sample questions right next to it. Instead, you want a "starfish" pattern: a set of questions that are all clearly connected to the central topic but extend in different semantic directions, covering the full range of the table's capabilities.

---

### The 4-Step Programmatic Workflow

This process can be built into a tool for your subject matter experts.

#### Step 1: Seed Generation from Schema

First, we need to programmatically generate "seed" topics based on the table's schema. This ensures our questions are grounded in the table's actual data capabilities.

*   **Action:** For a given table (e.g., `orders`), parse its `columns`. Create a list of topics.
    *   **Single-Column Topics:** "Questions about `order_date`", "Questions about `total_amount`".
    *   **Multi-Column Topics (for aggregations/grouping):** "Questions about `total_amount` grouped by `order_date`", "Questions about the count of orders per `customer_id`".

**Example for `orders` table:**
*   Seed 1: "Questions about order timing and dates."
*   Seed 2: "Questions about order value and revenue."
*   Seed 3: "Questions about sales performance over time."
*   Seed 4: "Questions about customer purchasing frequency."

#### Step 2: LLM-Powered Question Expansion

Now, use a powerful LLM to brainstorm a large pool of natural-language questions based on these seeds. This is where we get creativity and realistic phrasing.

*   **Action:** For each seed topic, send a prompt to an LLM.

**Example Prompt to LLM:**
```prompt
You are a business analyst brainstorming questions for a new dashboard.
The data comes from an 'orders' table with columns (id, customer_id, order_date, total_amount).

Based on the following topic, generate 5 distinct, natural-language questions a user might ask.

Topic: "Questions about sales performance over time."
```

**LLM Output (Candidate Pool):**
*   "What was our daily revenue last month?"
*   "Show me the sales trend for Q2."
*   "How did our total sales amount change week over week?"
*   "Compare the revenue from this year to last year."
*   "Is our overall order value growing or shrinking?"
*   *(...and many more from the other seed topics...)*

You now have a large candidate pool of 20-30 questions.

#### Step 3: Programmatic Diversification using Embeddings

This is the core of the solution. We will use an algorithm similar to **Maximal Marginal Relevance (MMR)** to select a small, diverse set from our large candidate pool.

*   **Action:**
    1.  Embed all candidate questions generated in Step 2.
    2.  Embed the table's core description (e.g., "Table: orders. Description: Records every sales transaction..."). This will be our "anchor" to ensure relevance.
    3.  Run the diversification selection algorithm.

**The Diversification Algorithm:**

1.  **Select the First Question:** Find the candidate question that is most similar (highest cosine similarity) to the table's core description. This becomes the first question in our final set. It's the most "on-topic" question.
2.  **Iteratively Select the Rest:** For each subsequent selection (until we have 5-7 questions):
    *   For every remaining candidate question, calculate two scores:
        *   **Relevance Score:** Its cosine similarity to the original table description.
        *   **Diversity Score:** Its similarity to the questions *already selected*. To get a single number, find the **maximum** similarity to any question already in our final set. A low score here is good (it means it's different).
    *   Calculate a final **MMR-like score** for each candidate: `Final Score = (lambda * Relevance) - ((1 - lambda) * MaxSimilarityToSelected)`
        *   `lambda` is a parameter (e.g., 0.7) that balances relevance vs. diversity.
    *   Select the candidate with the highest `Final Score` and add it to our final set. Repeat.

**Python Snippet for the Algorithm:**
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def select_diverse_questions(table_description: str, candidates: list, num_to_select: int = 5, lambda_val: float = 0.7):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Embed everything
    desc_embedding = model.encode([table_description])
    candidate_embeddings = model.encode(candidates)
    
    # Calculate relevance to the core description
    relevance_scores = cosine_similarity(desc_embedding, candidate_embeddings)[0]
    
    # Start the selection process
    selected_indices = []
    
    # 1. Select the first, most relevant question
    first_idx = np.argmax(relevance_scores)
    selected_indices.append(first_idx)
    
    remaining_indices = [i for i in range(len(candidates)) if i != first_idx]
    
    # 2. Iteratively select the rest
    while len(selected_indices) < num_to_select and remaining_indices:
        selected_embeddings = candidate_embeddings[selected_indices]
        
        best_candidate_idx = -1
        highest_mmr_score = -np.inf
        
        for idx in remaining_indices:
            relevance = relevance_scores[idx]
            
            # Calculate max similarity to already selected questions
            similarity_to_selected = cosine_similarity([candidate_embeddings[idx]], selected_embeddings)
            max_similarity = np.max(similarity_to_selected)
            
            # MMR-like calculation
            mmr_score = (lambda_val * relevance) - ((1 - lambda_val) * max_similarity)
            
            if mmr_score > highest_mmr_score:
                highest_mmr_score = mmr_score
                best_candidate_idx = idx
        
        if best_candidate_idx != -1:
            selected_indices.append(best_candidate_idx)
            remaining_indices.remove(best_candidate_idx)
            
    return [candidates[i] for i in selected_indices]

# --- Example Usage ---
# table_desc = "Table: orders. Description: Records every sales transaction..."
# candidate_pool = ["What was our daily revenue?", "How much did we sell yesterday?", "List our top customers by sales", ...]
# diverse_set = select_diverse_questions(table_desc, candidate_pool)
# print(diverse_set)
```

#### Step 4: Human Review and Curation

The final step is to present this programmatically selected, diverse set of 5-7 questions to the human expert.

*   **Action:** The HITL interface shows the recommended list.
*   **Expert's Role:**
    *   Final sanity check: Are these questions clear and relevant?
    *   Minor edits: Tweak the phrasing for clarity or business-specific terminology.
    *   Approve: Click "Save" to write these high-quality, diverse questions to the table's YAML file.

### The Workflow Diagram

```mermaid
graph TD
    A[1. Generate Seed Topics from Schema] --> B;
    B[2. LLM Expands Seeds into Large Candidate Pool] --> C;
    C[3. Programmatic Diversification (MMR Algorithm)] --> D{Selects 5-7 Diverse Questions};
    D --> E[4. Human Expert Review & Curation];
    E --> F[Save to YAML File];
    F --> G[Re-Index Graph];
```

This comprehensive approach ensures you're not just randomly adding questions. You are systematically building a small, powerful set of examples that covers the breadth of a table's purpose, leading to a much more robust and intelligent retrieval system.