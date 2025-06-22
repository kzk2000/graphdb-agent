import plotly.graph_objects as go
import plotly.io as pio
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

pio.renderers.default = "browser"


def visualize_question_diversity(table_description: str, sample_questions: list, title: str):
    """
    Generates an interactive 2D plot to visualize the semantic diversity
    of sample questions around a central table description.
    """
    print(f"--- Generating visualization for: {title} ---")

    # 1. Gather all texts
    all_texts = [table_description] + sample_questions

    # 2. Embed all texts
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_texts)

    # 3. Reduce dimensions using t-SNE
    # We are projecting 384 dimensions down to 2
    tsne = TSNE(n_components=2, perplexity=len(all_texts) - 1, random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings)

    # 4. Plot using Plotly
    # Separate the coordinates for the table and the questions
    table_coord = embeddings_2d[0]
    question_coords = embeddings_2d[1:]

    # Create the plot
    fig = go.Figure()

    # Add the question points (satellites)
    fig.add_trace(go.Scatter(
        x=question_coords[:, 0],
        y=question_coords[:, 1],
        mode='markers+text',
        name='Sample Questions',
        marker=dict(color='blue', size=12),
        text=[f"Q{i + 1}" for i in range(len(sample_questions))],  # Label points Q1, Q2, etc.
        textposition="top center",
        hoverinfo='text',
        hovertext=sample_questions  # Show full question text on hover
    ))

    # Add the table description point (the center)
    fig.add_trace(go.Scatter(
        x=[table_coord[0]],
        y=[table_coord[1]],
        mode='markers+text',
        name='Table Description',
        marker=dict(color='red', size=18, symbol='star'),
        text="Table",
        textposition="bottom center",
        hoverinfo='text',
        hovertext=[table_description]
    ))

    fig.update_layout(
        title=title,
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        showlegend=True,
        legend_title_text='Point Type'
    )

    fig.show()


# --- Example Usage ---
if __name__ == '__main__':
    # --- EXAMPLE 1: A GOOD, DIVERSE "SEMANTIC STARFISH" ---
    table_desc_orders = "Table: orders. Description: Records every sales transaction made by customers, including financial totals and dates."

    diverse_questions = [
        "What was our total revenue last month?",  # Financial, time-based
        "Which customers are our biggest spenders?",  # Financial, customer-based
        "Show me the sales trend for Q2.",  # Time-series analysis
        "How many orders did we process yesterday?",  # Simple count
        "List all orders over $1000 placed in January.",  # Complex filtering
        "What is the average order value per customer?"  # Aggregation
    ]

    visualize_question_diversity(table_desc_orders, diverse_questions, title="Good Diversity: The 'Semantic Starfish'")

    # --- EXAMPLE 2: A POOR, NON-DIVERSE "SEMANTIC CLUMP" ---
    table_desc_products = "Table: products. Description: Contains a catalog of all products available for sale."

    clumped_questions = [
        "List all our products.",
        "Show me every product we have.",
        "Can I see a list of all items?",
        "Give me the full product catalog.",
        "What products do we sell?",
        "Display all items for sale."
    ]

    visualize_question_diversity(table_desc_products, clumped_questions, title="Poor Diversity: The 'Semantic Clump'")
