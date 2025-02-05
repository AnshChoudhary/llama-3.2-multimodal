import gradio as gr
import PyPDF2
import numpy as np
from openai import OpenAI
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize OpenAI client with LiteLLM configuration
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),  # Your LiteLLM API key
    base_url="https://litellm.deriv.ai/v1"  # LiteLLM endpoint
)

# Configuration
LITELLM_MODEL = os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo')  # Default model, can be overridden by env var

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file.name)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text(text, chunk_size=1000):
    """Split text into chunks of approximately equal size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for space
        
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:  # Add the last chunk if it exists
        chunks.append(' '.join(current_chunk))
    
    return chunks

def find_most_relevant_chunk(query, chunks, vectorizer):
    """Find the most relevant text chunk for a given query."""
    query_vector = vectorizer.transform([query])
    chunk_vectors = vectorizer.transform(chunks)
    similarities = cosine_similarity(query_vector, chunk_vectors)
    most_relevant_idx = similarities.argmax()
    return chunks[most_relevant_idx]

class ChatBot:
    def __init__(self):
        self.messages = []
        self.chunks = None
        self.vectorizer = None
        self.faq_loaded = False
        self.system_prompt = """You are a friendly and helpful AI customer service assistant. 
        Provide clear, concise, and accurate responses based on the FAQ information provided. 
        If you're not sure about something, please say so."""

    def process_faq(self, pdf_file):
        if pdf_file is None:
            return "Please upload a PDF file."
        
        try:
            faq_text = extract_text_from_pdf(pdf_file)
            self.chunks = split_text(faq_text)
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(self.chunks)
            self.faq_loaded = True
            return "FAQ document loaded successfully!"
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def respond(self, message, history):
        if not self.faq_loaded:
            return "Please upload an FAQ document first."

        try:
            # Find most relevant chunk
            relevant_chunk = find_most_relevant_chunk(message, self.chunks, self.vectorizer)

            # Prepare messages for API
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""Based on the following FAQ information:
                
                {relevant_chunk}
                
                Customer question: {message}
                
                Please provide a helpful response."""}
            ]

            # Generate response using LiteLLM
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
        # Customer Support AI Assistant
        Using model: {LITELLM_MODEL}
        
        Upload your FAQ document (PDF) and start chatting!
        """)
        
        with gr.Row():
            pdf_upload = gr.File(label="Upload FAQ PDF", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
        
        pdf_upload.upload(
            fn=chatbot.process_faq,
            inputs=[pdf_upload],
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