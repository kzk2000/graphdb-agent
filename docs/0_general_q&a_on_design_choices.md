# Reasoning behind various design choices

### Q1: Which separator is best for LLMs. Is " ." good? Is "***" better? 
### A1:
That is an excellent and highly detail-oriented question. You've hit upon a subtle but important aspect of prompt engineering: **the choice of separator matters.**

You are absolutely right to question using `". "` as a joiner, as it can lead to awkward `..` punctuation, which can confuse the LLM's parser or at least look unprofessional.

Your suggestion of `***` is a good thought because it's visually distinct, but it's not ideal. The best separator is one that is **unambiguous, standard, and easily understood by both humans and LLMs** trained on trillions of tokens of text and code.

### The Recommendation: Use a Semicolon (`;`) or Vertical Bar (`|`)

The two best choices for this are the **semicolon (`;`)** or the **vertical bar (`|`)**.

Here’s a comparison:

| Separator               | LLM Clarity  | Human Readability | Why it's a good choice |
|:------------------------|:-------------| :--- | :--- |
| **Semicolon (`;`)**     | **Excellent** | **Excellent** | This is the classic, grammatically correct way to separate distinct but related clauses or items in a list. The LLM has seen this pattern billions of times and understands it perfectly as a strong separator. |
| **Vertical Bar (`\|`)** | **Excellent** | **Good** | This is a very common separator in technical contexts, data formats, and documentation. It's visually clean and signals a clear, unambiguous separation of concepts. The LLM will also interpret this as a very strong separator. |
| Asterisks (`***`)       | Good         | Okay | This is non-standard punctuation. While an LLM can likely infer the meaning, it's "noisier" and less conventional. It might be interpreted as emphasis or a markdown artifact rather than a pure separator. |

**Verdict:** The **semicolon (`;`)** is arguably the best choice as it strikes a perfect balance between being a standard grammatical separator and a clear logical one. The vertical bar (`|`) is an equally strong technical choice.



That's a fantastic question that gets into the fine details of prompt design and information hierarchy.

The answer is: **No, you should not use the semicolon (`;`) as a separator *within* the list of rules.**

Using the same separator for different levels of information can create ambiguity and make it harder for the LLM to parse the structure correctly.

#### The Principle: Hierarchical Separators

Think of your metadata comment as having a structure:

*   **Level 1 (Top Level):** Separating the main categories of metadata.
    *   `Description` ; `Synonyms` ; `Rules`
*   **Level 2 (Nested Level):** Separating the individual items *within* a category.
    *   Within `Synonyms`: `revenue`, `sales`, `income`
    *   Within `Rules`: `Rule A`, `Rule B`, `Rule C`

You should use a **strong, primary separator** for Level 1 (we chose the semicolon `;`) and a **weaker, secondary separator** for Level 2.

### Why Using Semicolons Everywhere is Bad

Let's look at what the LLM would see if you used semicolons for both levels:

**Ambiguous Prompt (Bad):**
```sql
-- ... ; Rules: The value 'Shipped' means...; The value 'Delivered' means...; Use `status = 'Shipped'` for...
```
When the LLM parses this, it sees a flat list of semi-colon separated items. It loses the clear grouping that all three statements after "Rules:" belong together under that single heading. It might misinterpret the second rule as a new top-level category.

### The Best Practice: Treat Rules as a Coherent Block of Text

The best way to handle the list of rules is to treat them as a set of complete, natural language sentences. This is a format that LLMs are exceptionally good at understanding.

