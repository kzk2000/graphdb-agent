import spacy
from typing import List, Dict

NLP = spacy.load("en_core_web_sm")


def lemmatize_text(text: str) -> List[str]:
    """
    Processes text using spaCy to lemmatize and remove stop words and punctuation.
    """

    doc = NLP(text.lower())
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return lemmas


def key_word_gap_analysis(user_query, table_metadata_text):
    """
    Keyword Gap Analysis
    This is the most direct and essential strategy. The program identifies what words are in the query but missing from the table's metadata.
    Concept: Programmatically find the semantic "missing link" and present it to the expert.

    Workflow:
    The HITL interface shows the query and the expert's correct table choice (suppliers).
    The program lemmatizes both the user query and all the text associated with the suppliers table (name, description, columns, synonyms).
    It performs a set difference: keywords_in_query - keywords_in_table_metadata.
    The result is presented to the expert as a "suggestion."

    Example Interaction:
    System to Expert:
    The user asked about "total revenue from our top partners".
    You selected the suppliers table as the correct choice.
    Suggestion: The suppliers table metadata does not contain the important concepts: 'revenue', 'partner'.
    Consider adding these concepts to the table's description, sample questions, or as a synonym for a relevant column.
    """

    if False:
        # example for testing
        user_query = "What is the total revenue from our top partners?"
        table_metadata_text = "suppliers Lists all vendors and companies that supply..."

    query_lemmas = set(lemmatize_text(user_query))
    table_lemmas = set(lemmatize_text(table_metadata_text))

    missing_keywords = query_lemmas - table_lemmas
    # missing_keywords would be {'revenue', 'partner'}

    print(f"Suggestion: Add these concepts: {missing_keywords}")
