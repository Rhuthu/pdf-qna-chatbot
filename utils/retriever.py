import faiss
import numpy as np

def build_faiss_index(embeddings):
    """
    Builds a FAISS index from the given embeddings.
    
    Args:
        embeddings (List[List[float]]): List of dense vectors.
        
    Returns:
        faiss.IndexFlatL2: Trained FAISS index
    """
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def search_index(index, query_embedding, top_k=3):
    """
    Search the FAISS index for the top_k most similar vectors.
    
    Args:
        index: FAISS index
        query_embedding: embedding of the query string
        top_k: number of top results to return
        
    Returns:
        List of (index, score) tuples
    """
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]
