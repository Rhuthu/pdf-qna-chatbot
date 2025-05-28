def chunk_text(text, max_length=500, overlap=50):
    """
    Splits text into chunks of max_length with overlap.
    
    Args:
        text (str): The text to split.
        max_length (int): Maximum length of each chunk.
        overlap (int): Overlap size between chunks.
        
    Returns:
        list[str]: List of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_length - overlap  # move start forward for overlap
    
    return chunks
