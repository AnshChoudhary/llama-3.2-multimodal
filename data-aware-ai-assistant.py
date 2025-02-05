import gradio as gr
import PyPDF2
import numpy as np
from openai import OpenAI
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import io
from typing import List, Union
import openpyxl

# Initialize OpenAI client with LiteLLM configuration
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),  # Your LiteLLM API key
    base_url="https://litellm.deriv.ai/v1"  # LiteLLM endpoint
)

# Configuration
LITELLM_MODEL = os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo')  # Default model, can be overridden by env var

def extract_text_from_file(file) -> str:
    """Extract text from PDF or spreadsheet file."""
    file_ext = file.name.split('.')[-1].lower()
    
    if file_ext == 'pdf':
        pdf_reader = PyPDF2.PdfReader(file.name)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    elif file_ext in ['xlsx', 'xls']:
        df = pd.read_excel(file.name)
        return process_dataframe(df)
    
    elif file_ext == 'csv':
        df = pd.read_csv(file.name)
        return process_dataframe(df)
    
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

def process_dataframe(df: pd.DataFrame) -> str:
    """Convert DataFrame to structured text format."""
    # Drop empty columns and rows
    df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)
    
    # Convert DataFrame to structured text
    text_chunks = []
    
    # Add column names as context
    columns_desc = "Columns: " + ", ".join(df.columns.astype(str))
    text_chunks.append(columns_desc)
    
    # Process each row
    for idx, row in df.iterrows():
        # Create a structured text representation of the row
        row_text = f"Entry {idx + 1}:\n"
        for col in df.columns:
            value = row[col]
            if pd.notna(value):  # Only include non-null values
                row_text += f"{col}: {value}\n"
        text_chunks.append(row_text)
    
    return "\n\n".join(text_chunks)

def split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks with improved handling of structured data."""
    # Split by double newlines to preserve data structure
    sections = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for section in sections:
        section_size = len(section)
        
        # If a single section is larger than chunk_size, split it by single newlines
        if section_size > chunk_size:
            lines = section.split('\n')
            for line in lines:
                if current_size + len(line) > chunk_size:
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        # Keep last few lines for overlap
                        current_chunk = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                        current_size = sum(len(line) + 1 for line in current_chunk)
                current_chunk.append(line)
                current_size += len(line) + 1
        else:
            if current_size + section_size > chunk_size:
                chunks.append('\n'.join(current_chunk))
                # Keep last entry for overlap
                current_chunk = current_chunk[-1:] if current_chunk else []
                current_size = sum(len(line) + 1 for line in current_chunk)
            current_chunk.append(section)
            current_size += section_size + 2  # +2 for double newline
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

class ChatBot:
    def __init__(self):
        self.messages = []
        self.chunks = None
        self.vectorizer = None
        self.faq_loaded = False
        self.system_prompt = """You are a helpful AI assistant specializing in understanding and explaining data from documents and spreadsheets. 
        When providing information:
        1. Be precise and accurate with numbers and facts
        2. Maintain context about columns and data structure
        3. If multiple relevant entries exist, summarize them clearly
        4. If you're unsure about something, say so
        5. Format numerical data clearly and consistently"""

    def process_faq(self, file):
        if file is None:
            return "Please upload a file."
        
        try:
            # Extract text using the enhanced extraction function
            content_text = extract_text_from_file(file)
            
            # Create chunks with the improved chunking strategy
            self.chunks = split_text(content_text)
            
            # Initialize and fit the vectorizer with stop words removed
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                token_pattern=r'(?u)\b\w+\b',  # Include single-character words
                ngram_range=(1, 2)  # Include bigrams for better context
            )
            self.vectorizer.fit(self.chunks)
            
            self.faq_loaded = True
            return f"Document processed successfully! Found {len(self.chunks)} sections."
        except Exception as e:
            return f"Error processing file: {str(e)}"

    def find_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Find the most relevant chunks for a given query."""
        query_vector = self.vectorizer.transform([query])
        chunk_vectors = self.vectorizer.transform(self.chunks)
        similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        
        # Get top k chunks
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices if similarities[i] > 0.1]

    def respond(self, message, history):
        if not self.faq_loaded:
            return "Please upload a document first."

        try:
            # Find multiple relevant chunks
            relevant_chunks = self.find_relevant_chunks(message)
            
            if not relevant_chunks:
                return "I couldn't find relevant information to answer your question. Please try rephrasing it."

            # Combine relevant chunks with context
            context = "\n\n".join(relevant_chunks)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""Based on the following information:

{context}

Question: {message}

Please provide a helpful response, citing specific data where relevant."""}
            ]

            response = client.chat.completions.create(
                model=LITELLM_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content

        except Exception as e:
            return f"Error generating response: {str(e)}"

def create_demo():
    chatbot = ChatBot()
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="zinc")) as demo:
        gr.Markdown(f"""
        # Data-Aware AI Assistant
        Using model: {LITELLM_MODEL}
        
        Upload your document (PDF, Excel, or CSV) and start chatting!
        """)
        
        with gr.Row():
            file_upload = gr.File(label="Upload Document", file_types=[".pdf", ".xlsx", ".xls", ".csv"])
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
        
        file_upload.upload(
            fn=chatbot.process_faq,
            inputs=[file_upload],
            outputs=[upload_status]
        )
        
        chatbot_interface = gr.ChatInterface(
            chatbot.respond,
            chatbot=gr.Chatbot(height=400),
            title="Chat with our AI Assistant",
            description="Ask any questions about our services!",
        )

    return demo

if __name__ == "__main__":
    # Verify required environment variables
    required_vars = {
        'OPENAI_API_KEY': 'your LiteLLM API key',
        'OPENAI_MODEL_NAME': 'the model name you want to use (optional)'
    }
    
    missing_vars = [var for var, desc in required_vars.items() if not os.getenv(var) and var != 'OPENAI_MODEL_NAME']
    
    if missing_vars:
        print("\nMissing required environment variables:")
        for var in missing_vars:
            print(f"- {var}: Set this to {required_vars[var]}")
        print("\nYou can set these using:")
        print('export OPENAI_API_KEY="your-litellm-api-key"')
        print('export OPENAI_MODEL_NAME="your-chosen-model"  # Optional')
    else:
        demo = create_demo()
        demo.launch()