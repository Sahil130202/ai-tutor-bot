# AI Tutor Bot using GPT-4 + LangChain
# Complete implementation with PDF processing, web search, and conversational AI

import os
import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.tools import DuckDuckGoSearchRun
import tempfile
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Any
import pickle
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AITutorBot:
    def __init__(self, openai_api_key: str):
        """Initialize the AI Tutor Bot with OpenAI API key."""
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Initialize core components
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        
        # Initialize vector store
        self.vectorstore = None
        self.conversation_chain = None
        
        # Initialize web search
        self.web_search = DuckDuckGoSearchRun()
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            memory_key="chat_history",
            return_messages=True
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        logger.info("AI Tutor Bot initialized successfully")

    def load_pdf_documents(self, pdf_files: List[Any]) -> List[Document]:
        """Load and process PDF documents."""
        documents = []
        
        for pdf_file in pdf_files:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_file_path = tmp_file.name
                
                # Load PDF
                loader = PyPDFLoader(tmp_file_path)
                pdf_documents = loader.load()
                
                # Add metadata
                for doc in pdf_documents:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["type"] = "pdf"
                
                documents.extend(pdf_documents)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                logger.info(f"Loaded PDF: {pdf_file.name} with {len(pdf_documents)} pages")
                
            except Exception as e:
                logger.error(f"Error loading PDF {pdf_file.name}: {str(e)}")
                st.error(f"Error loading PDF {pdf_file.name}: {str(e)}")
        
        return documents

    def create_vector_store(self, documents: List[Document]):
        """Create vector store from documents."""
        if not documents:
            st.warning("No documents provided for vector store creation")
            return
        
        try:
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(texts)} text chunks from {len(documents)} documents")
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            logger.info("Vector store created successfully")
            st.success(f"Processed {len(documents)} documents into {len(texts)} chunks")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            st.error(f"Error creating vector store: {str(e)}")

    def search_web(self, query: str, num_results: int = 3) -> str:
        """Search the web for additional information."""
        try:
            search_results = self.web_search.run(query)
            logger.info(f"Web search completed for query: {query}")
            return search_results
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return f"Web search unavailable: {str(e)}"

    def create_conversation_chain(self):
        """Create the conversational retrieval chain."""
        if not self.vectorstore:
            st.error("Please upload and process documents first!")
            return
        
        # Custom prompt template
        custom_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""You are an AI Tutor Bot designed to help students learn effectively. 
            Use the following context from documents and web search to answer the question.
            
            Context from documents and web:
            {context}
            
            Chat History:
            {chat_history}
            
            Question: {question}
            
            Instructions:
            1. Provide clear, educational explanations
            2. Use examples when helpful
            3. If the context doesn't contain enough information, mention what additional topics the student might want to explore
            4. Be encouraging and supportive
            5. Break down complex concepts into simpler parts
            
            Answer:"""
        )
        
        # Create retrieval chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
        
        logger.info("Conversation chain created successfully")

    def get_response(self, question: str) -> Dict[str, Any]:
        """Get response from the AI tutor."""
        if not self.conversation_chain:
            return {
                "answer": "Please upload and process documents first!",
                "sources": []
            }
        
        try:
            # First, search web for additional context
            web_context = self.search_web(question)
            
            # Enhance question with web context
            enhanced_question = f"{question}\n\nAdditional web context: {web_context}"
            
            # Get response from conversation chain
            response = self.conversation_chain({
                "question": enhanced_question
            })
            
            return {
                "answer": response["answer"],
                "sources": response.get("source_documents", []),
                "web_context": web_context
            }
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": []
            }

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AI Tutor Bot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Tutor Bot")
    st.markdown("### Your Personal AI Learning Assistant")

    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key input
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use GPT-4"
        )
        
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key to continue")
            st.stop()
        
        # Initialize the bot
        if "tutor_bot" not in st.session_state:
            st.session_state.tutor_bot = AITutorBot(openai_api_key)
        
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF documents that the AI will use to answer your questions"
        )
        
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    documents = st.session_state.tutor_bot.load_pdf_documents(uploaded_files)
                    st.session_state.tutor_bot.create_vector_store(documents)
                    st.session_state.tutor_bot.create_conversation_chain()
                st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one PDF document")
        
        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.tutor_bot.clear_memory()
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.success("Conversation cleared!")
    
    # Main chat interface
    st.header("💬 Chat with Your AI Tutor")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}:**")
                        st.write(f"- Document: {source.metadata.get('source', 'Unknown')}")
                        st.write(f"- Page: {source.metadata.get('page', 'Unknown')}")
                        st.write(f"- Content: {source.page_content[:200]}...")
                        st.write("---")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.tutor_bot.get_response(prompt)
                
                st.markdown(response["answer"])
                
                # Display sources
                if response.get("sources"):
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(response["sources"]):
                            st.write(f"**Source {i+1}:**")
                            st.write(f"- Document: {source.metadata.get('source', 'Unknown')}")
                            st.write(f"- Page: {source.metadata.get('page', 'Unknown')}")
                            st.write(f"- Content: {source.page_content[:200]}...")
                            st.write("---")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": response.get("sources", [])
                })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**AI Tutor Bot** - Powered by GPT-4 and LangChain | "
        "Upload your study materials and start learning!"
    )

if __name__ == "__main__":
    main()

# Additional utility functions and classes

class DocumentProcessor:
    """Advanced document processing utilities."""
    
    @staticmethod
    def extract_text_from_url(url: str) -> str:
        """Extract text content from a webpage."""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from URL {url}: {str(e)}")
            return ""
    
    @staticmethod
    def save_chat_history(messages: List[Dict], filename: str = None):
        """Save chat history to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.pkl"
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(messages, f)
            logger.info(f"Chat history saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
    
    @staticmethod
    def load_chat_history(filename: str) -> List[Dict]:
        """Load chat history from file."""
        try:
            with open(filename, 'rb') as f:
                messages = pickle.load(f)
            logger.info(f"Chat history loaded from {filename}")
            return messages
        except Exception as e:
            logger.error(f"Error loading chat history: {str(e)}")
            return []

# Performance optimization utilities
class CacheManager:
    """Manage caching for improved performance."""
    
    def __init__(self):
        self.cache = {}
    
    def get(self, key: str):
        """Get item from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set item in cache with TTL."""
        self.cache[key] = {
            'value': value,
            'timestamp': datetime.now().timestamp(),
            'ttl': ttl
        }
    
    def is_valid(self, key: str) -> bool:
        """Check if cached item is still valid."""
        if key not in self.cache:
            return False
        
        item = self.cache[key]
        return (datetime.now().timestamp() - item['timestamp']) < item['ttl']

# Configuration class
class Config:
    """Configuration settings for the AI Tutor Bot."""
    
    # Model settings
    MODEL_NAME = "gpt-4"
    TEMPERATURE = 0.7
    MAX_TOKENS = 2000
    
    # Chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval settings
    TOP_K_DOCUMENTS = 4
    
    # Memory settings
    MEMORY_WINDOW = 5
    
    # Web search settings
    WEB_SEARCH_RESULTS = 3
    
    # Vector store settings
    VECTOR_STORE_DIR = "./chroma_db"