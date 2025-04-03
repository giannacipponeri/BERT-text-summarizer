# summarizer.py
import os
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources are downloaded
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Check if punkt is available, download if not
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt tokenizer...")
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)

# Load BERT model and tokenizer
print("Loading BERT model...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

def get_bert_embeddings(sentences, batch_size=8):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:min(i+batch_size, len(sentences))]
        if not batch or all(not s.strip() for s in batch):
            continue
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings

def summarize_text(text, num_sentences=3):
    if not text or not text.strip():
        return "No text provided for summarization."
        
    # Double-check that punkt is available before tokenizing
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # One more attempt to download punkt
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        sentences = sent_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    valid_sentences = []
    valid_indices = []
    for i, sentence in enumerate(sentences):
        if len(sentence.split()) > 3:
            valid_sentences.append(sentence)
            valid_indices.append(i)

    if len(valid_sentences) <= num_sentences:
        return ' '.join(valid_sentences)

    embeddings = get_bert_embeddings(valid_sentences)
    position_scores = [1.0 / (i + 1) for i in valid_indices]
    similarity_matrix = cosine_similarity(embeddings)

    sentence_scores = []
    for i in range(len(valid_sentences)):
        content_score = np.mean(similarity_matrix[i])
        pos_score = position_scores[i]
        length = len(valid_sentences[i].split())
        length_score = min(length / 20, 1.0) if length < 20 else 20 / length
        final_score = 0.4 * content_score + 0.5 * pos_score + 0.1 * length_score
        sentence_scores.append((i, final_score))

    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    selected_indices = [valid_indices[idx] for idx, _ in top_sentences]
    selected_indices.sort()
    summary = ' '.join([sentences[i] for i in selected_indices])
    return summary