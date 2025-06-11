# AI Tutor Bot: Project Documentation

## Overview

AI Tutor Bot is an intelligent conversational assistant that helps users understand PDF-based study material by combining the power of OpenAI's GPT-4 with LangChain's Retrieval-Augmented Generation (RAG) capabilities. It features document ingestion, contextual Q&A, chat memory, and optional web search to supplement knowledge.

---

## System Architecture

```
┌─────────────┐        ┌─────────────┐        ┌────────────────────────┐
│   User UI   │<──────>│ Streamlit UI│<──────>│     AI Tutor Backend    │
└─────────────┘        └─────────────┘        └────────────────────────┘
                                                │
                   ┌────────────────────────────┼───────────────────────┐
                   ▼                            ▼                       ▼
           Vector Store (Chroma)      GPT-4 via LangChain         Web Search (DuckDuckGo)
```

---

## Project Structure

```
ai_tutor_bot/
├── ai_tutor_bot.py              # Main Streamlit application
├── tutor/
│   ├── ai_tutor.py              # AI logic and orchestration
│   ├── processor.py             # Document/web handling utilities
│   ├── config.py                # Model and system config
├── chroma_db/                   # Persistent vector database
├── .gitignore
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```

---

##  How It Works

### 1. PDF Ingestion

- Users upload one or more PDF files.
- Each PDF is read using `PyPDFLoader`.
- Content is split into chunks using `RecursiveCharacterTextSplitter`.
- Chunks are embedded via `OpenAIEmbeddings` and stored in `Chroma`.

### 2. Conversational Retrieval

- User enters a question.
- LangChain's `ConversationalRetrievalChain`:
  - Retrieves top-k chunks using similarity search.
  - Includes recent chat history via `ConversationBufferWindowMemory`.
  - Feeds everything into `ChatOpenAI` (GPT-4).

### 3. Optional Web Search

- If enabled, DuckDuckGo is queried to fetch external information.
- Retrieved snippets are included in the prompt for GPT-4.

### 4. Response Rendering

- Answer is displayed in Streamlit's chat interface.
- Sources are shown with document name, page number, and preview.

---

##  Key Components

### `AITutorBot`

Handles:

- Document loading
- Vector store creation
- Response generation via LangChain chains
- Memory management
- Web search integration

### `DocumentProcessor`

Handles:

- Web scraping using `requests` + `BeautifulSoup`
- Saving and loading chat history with `pickle`

### `CacheManager`

- Lightweight in-memory cache with TTL

### `config.py`

Stores:

- Model settings (e.g. temperature, chunk size)
- Retrieval parameters

---

##  Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-tutor-bot.git
cd ai-tutor-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API Key

You can either:

- Enter it in the Streamlit sidebar
- Or set it in an environment variable:

```bash
export OPENAI_API_KEY="your-key-here"
```

### 4. Run the app

```bash
streamlit run ai_tutor_bot.py
```

---

##  Example Diagrams

###  PDF → Vectors

```
PDF ➝ Text ➝ Chunks ➝ Embeddings ➝ Chroma
```

###  Chat Flow

```
User Query ➝ Retrieve Chunks ➝ Combine with History ➝ GPT-4 ➝ Response
```

---

##  requirements.txt (core)

```
openai
langchain
langchain-community
streamlit
chromadb
duckduckgo-search
beautifulsoup4
```

---

##  Future Improvements

- Add support for other file types (.docx, .txt)
- Improve long-context handling using GPT-4 Turbo
- Enable user authentication for saving sessions
- Stream output from GPT-4 (real-time typing effect)

---

##  Credits

Created by Sahil Nayak Inspired by LangChain + OpenAI + the desire to make learning accessible through AI.

