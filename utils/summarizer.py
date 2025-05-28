from transformers import pipeline

# Load summarizer model globally to avoid reloading each time
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def better_summarize_pdf(text, chunk_size=1500, overlap=200):
    summaries = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        try:
            chunk_length = len(chunk.split())
            max_len = min(200, int(chunk_length * 0.8))
            min_len = max(30, int(chunk_length * 0.3))

            summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

        except Exception as e:
            summary = "[Summary failed]"
        summaries.append(summary)
        start += chunk_size - overlap

    combined_summary = " ".join(summaries)
    
    try:
        final = summarizer(combined_summary, max_length=250, min_length=100, do_sample=False)[0]['summary_text']
    except Exception:
        final = combined_summary[:1500]  # Fallback

    return final
