from sentence_transformers import SentenceTransformer

model=SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    """
     Generates embeddings for each text chunk.
    
    Args:
        chunks (list[str]): List of text chunks.
    
    Returns:
        list[list[float]]: List of embeddings (vectors).
    """

    embeddings= model.encode(chunks, show_progress_bar=True)
    return embeddings