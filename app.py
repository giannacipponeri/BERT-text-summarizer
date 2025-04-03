import streamlit as st
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import time

# Set page configuration
st.set_page_config(
    page_title="BERT News Summarizer",
    page_icon="📰",
    layout="wide"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_resources()

# Load BERT model and tokenizer - Using a smaller model for Streamlit Cloud
@st.cache_resource
def load_bert_model():
    with st.spinner("Loading BERT model... (this may take a minute on first run)"):
        # Using a smaller model than bert-base-uncased for better performance on Streamlit Cloud
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # Move to CPU as Streamlit Cloud likely doesn't have GPU
        device = torch.device("cpu")
        model = model.to(device)
        
        return tokenizer, model, device

# Function to get BERT embeddings for sentences
def get_bert_embeddings(sentences, tokenizer, model, device, batch_size=4):
    """
    Extract BERT embeddings for a list of sentences
    """
    embeddings = []
    
    # Process in smaller batches to avoid memory issues
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:min(i+batch_size, len(sentences))]
        
        # Skip empty sentences
        if not batch or all(not s.strip() for s in batch):
            continue
            
        # Tokenize with smaller max_length
        inputs = tokenizer(batch, padding=True, truncation=True, 
                          return_tensors="pt", max_length=256).to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Use [CLS] token embedding as sentence representation
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
    
    return embeddings

# Function to summarize using BERT and position-based approach
def bert_summarize(text, tokenizer, model, device, num_sentences=3, with_details=False):
    """
    Generate extractive summary using BERT embeddings and position weighting
    """
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Handle short articles
    if len(sentences) <= num_sentences:
        if with_details:
            return ' '.join(sentences), list(range(len(sentences))), []
        else:
            return ' '.join(sentences)
    
    # Filter out very short sentences
    valid_sentences = []
    valid_indices = []
    for i, sentence in enumerate(sentences):
        if len(sentence.split()) > 3:  # Skip very short sentences
            valid_sentences.append(sentence)
            valid_indices.append(i)
    
    # Handle case with too few valid sentences
    if len(valid_sentences) <= num_sentences:
        if with_details:
            return ' '.join(valid_sentences), valid_indices, []
        else:
            return ' '.join(valid_sentences)
    
    # Get BERT embeddings
    embeddings = get_bert_embeddings(valid_sentences, tokenizer, model, device)
    
    # Calculate position importance scores (earlier sentences more important in news)
    position_scores = [1.0 / (i + 1) for i in valid_indices]
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Calculate sentence scores
    sentence_scores = []
    for i in range(len(valid_sentences)):
        # Content score: how similar this sentence is to the overall document
        content_score = np.mean(similarity_matrix[i])
        
        # Position score: earlier sentences are more important in news
        pos_score = position_scores[i]
        
        # BERT-based length penalty (favor medium-length sentences)
        length = len(valid_sentences[i].split())
        length_score = min(length / 20, 1.0) if length < 20 else 20 / length
        
        # Combine scores with different weights
        final_score = 0.4 * content_score + 0.5 * pos_score + 0.1 * length_score
        
        sentence_scores.append((i, final_score))
    
    # Select top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    
    # Sort by original position to maintain coherence
    selected_indices = [valid_indices[idx] for idx, _ in top_sentences]
    original_indices = selected_indices.copy()
    selected_indices.sort()
    
    # Construct summary
    summary = ' '.join([sentences[i] for i in selected_indices])
    
    if with_details:
        # Return with details for highlighting
        return summary, selected_indices, sentence_scores
    else:
        return summary

# Main app
def main():
    st.title("📰 News Article Summarizer")
    st.subheader("Powered by BERT")
    
    # Add information about your project
    st.markdown("""
    **Course Project for Natural Language Processing**
    
    This application uses BERT embeddings to create extractive summaries of news articles. 
    The algorithm computes sentence importance based on semantic similarity and position.
    """)
    
    # Loading the model (with a loading indicator)
    with st.spinner("Loading DistilBERT model - this may take 1-2 minutes on first run..."):
        try:
            tokenizer, model, device = load_bert_model()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    # Sidebar options
    st.sidebar.header("Summarization Settings")
    top_n = st.sidebar.slider("Number of sentences to include", 1, 10, 3)
    show_highlights = st.sidebar.checkbox("Highlight selected sentences in original text", True)
    
    # Sample texts
    st.sidebar.header("Sample Articles")
    sample_articles = {
        "Technology News": """Apple unveiled its new iPhone model today at their annual developer conference in Cupertino. The iPhone 15 Pro features a revolutionary AI chip that allows for enhanced photography and real-time language translation. CEO Tim Cook called it "the most advanced iPhone we've ever created." The device will retail starting at $999 and will be available in stores next month. Analysts predict strong sales despite the higher price point compared to competitors.""",
        "Sports News": """The Boston Celtics defeated the Miami Heat 112-105 in Game 7 of the Eastern Conference Finals. Jayson Tatum led all scorers with 32 points, while Jaylen Brown added 27 points and 8 rebounds. The Celtics overcame an early 15-point deficit with a strong third quarter in which they outscored the Heat 34-21. "We never gave up and kept fighting," said Tatum after the game. The Celtics will now face the Denver Nuggets in the NBA Finals, which begin next Thursday."""
    }
    
    sample_choice = st.sidebar.radio("Try a sample article:", ["None"] + list(sample_articles.keys()))
    
    # Text input area
    if sample_choice != "None":
        article_text = sample_articles[sample_choice]
    else:
        article_text = st.text_area("Paste or type your article here:", height=300)
    
    # Process text when user inputs something
    if article_text:
        if st.button("Generate Summary"):
            with st.spinner("Analyzing with DistilBERT and generating summary..."):
                start_time = time.time()
                
                try:
                    # Generate summary with details for visualization
                    summary, selected_indices, sentence_scores = bert_summarize(
                        article_text, 
                        tokenizer, 
                        model, 
                        device,
                        top_n=top_n,
                        with_details=True
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Article")
                        if show_highlights and selected_indices:
                            # Highlight selected sentences in the original text
                            sentences = sent_tokenize(article_text)
                            
                            highlighted_text = ""
                            for i, sentence in enumerate(sentences):
                                if i in selected_indices:
                                    highlighted_text += f"<mark>{sentence}</mark> "
                                else:
                                    highlighted_text += f"{sentence} "
                            
                            st.markdown(highlighted_text, unsafe_allow_html=True)
                        else:
                            st.write(article_text)
                    
                    with col2:
                        st.subheader("Generated Summary")
                        st.success(summary)
                        
                        # Calculate and display stats
                        original_word_count = len(article_text.split())
                        summary_word_count = len(summary.split())
                        reduction = (1 - summary_word_count / original_word_count) * 100
                        
                        st.metric(
                            "Compression Rate", 
                            f"{reduction:.1f}%", 
                            f"From {original_word_count} to {summary_word_count} words"
                        )
                        
                        st.info(f"Processing time: {processing_time:.2f} seconds")
                
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
    
    # Information about the model
    with st.expander("About this summarizer"):
        st.markdown("""
        This summarizer uses DistilBERT (a lighter version of BERT) to create extractive summaries. The approach works as follows:
        
        1. The article is split into sentences
        2. DistilBERT generates embeddings for each sentence
        3. A scoring system combines:
           - Semantic similarity between sentences
           - Position in the document (earlier sentences often contain key information in news)
           - Optimal sentence length
        4. The top-scoring sentences are selected and arranged in their original order
        
        DistilBERT understands context and semantic meaning, allowing for more nuanced summarization than purely statistical approaches.
        """)

if __name__ == "__main__":
    main()