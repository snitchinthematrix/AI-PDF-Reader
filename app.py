from collections.abc import MutableMapping
import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer
import re

class LocalPDFReader:
    def __init__(self):
        # Initialize local models
        # Using smaller models that can run efficiently on CPU
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # Run on CPU
        )
        
        # Load BERT model for embeddings and similarity
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load small QA model
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=-1  # Run on CPU
        )

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def preprocess_text(self, text):
        """Clean and preprocess the extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def split_into_chunks(self, text, chunk_size=500):
        """Split text into manageable chunks"""
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            if len(sentence.strip()) == 0:
                continue
                
            current_chunk.append(sentence + '.')
            current_size += len(sentence)
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def analyze_rhetoric(self, text_chunk):
        """Analyze rhetorical elements using local models"""
        # Get sentiment
        sentiment = self.sentiment_analyzer(text_chunk)[0]
        
        # Get key sentences using embeddings
        sentences = text_chunk.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        
        if sentences:
            embeddings = self.embedding_model.encode(sentences)
            # Calculate sentence importance based on similarity to the whole text
            text_embedding = self.embedding_model.encode(text_chunk)
            importance_scores = torch.nn.functional.cosine_similarity(
                torch.tensor(text_embedding).unsqueeze(0),
                torch.tensor(embeddings),
                dim=1
            )
            
            # Get top 3 important sentences
            top_indices = importance_scores.argsort(descending=True)[:3]
            key_points = [sentences[idx] for idx in top_indices]
        else:
            key_points = []

        return {
            'sentiment': sentiment,
            'key_points': key_points
        }

    def answer_question(self, question, context):
        """Answer questions about the text using local model"""
        try:
            answer = self.qa_pipeline(question=question, context=context)
            return answer['answer']
        except Exception as e:
            return f"I couldn't find a specific answer to that question. Try rephrasing it?"

def main():
    st.title("PDF Reader & Analysis")
    st.write("Upload a PDF to analyze its content (100% local - no API keys needed!)")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Initialize reader
        reader = LocalPDFReader()
        
        # Extract and process text
        with st.spinner("Processing PDF..."):
            text = reader.extract_text_from_pdf(uploaded_file)
            processed_text = reader.preprocess_text(text)
            chunks = reader.split_into_chunks(processed_text)
        
        # Display text
        st.subheader("Document Text")
        chunk_selector = st.selectbox(
            "Select text chunk to analyze:",
            range(len(chunks)),
            format_func=lambda x: f"Chunk {x+1}"
        )
        st.text_area("Text Content", chunks[chunk_selector], height=200)
        
        # Analysis section
        if st.button("Analyze This Chunk"):
            with st.spinner("Analyzing..."):
                analysis = reader.analyze_rhetoric(chunks[chunk_selector])
                
                st.subheader("Analysis Results")
                st.write("Sentiment:", analysis['sentiment']['label'])
                st.write("Confidence:", f"{analysis['sentiment']['score']:.2%}")
                
                st.write("Key Points:")
                for idx, point in enumerate(analysis['key_points'], 1):
                    st.write(f"{idx}. {point}")
        
        # Question answering section
        st.subheader("Ask Questions")
        question = st.text_input("Ask a question about the text:")
        
        if question and st.button("Get Answer"):
            with st.spinner("Finding answer..."):
                answer = reader.answer_question(question, chunks[chunk_selector])
                st.write("Answer:", answer)

if __name__ == "__main__":
    main()