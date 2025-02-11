# RAG Q&A Conversational with PDF including Chat History

This Streamlit application allows users to upload PDFs and interact with their content using a **Retrieval-Augmented Generation (RAG) pipeline**. The app maintains chat history for a better conversational experience and uses **Groq's LLM (Gemma2-9b-It)** for generating responses.

## Features
- 📄 **PDF Upload**: Users can upload PDF files for text extraction.
- 🔍 **RAG Pipeline**: Uses a retriever and a context-aware LLM for generating answers.
- 🧠 **Chat History**: Maintains context over multiple interactions.
- 🔑 **API Key Support**: Uses Groq API Key for authentication.
- 🏗 **Chroma Vector Database**: Stores and retrieves document chunks efficiently.

## Installation

### Prerequisites
- Python 3.8+
- [Streamlit](https://streamlit.io/)
- Groq API Key (sign up at [Groq](https://groq.com/))
- HuggingFace API Token (for embeddings)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # For macOS/Linux
   env\Scripts\activate     # For Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project directory and add:
   ```env
   GROQ_API_KEY=your_groq_api_key
   HF_TOKEN=your_huggingface_token
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Enter your Groq API Key** in the text input field.
2. **Upload PDF files** that you want to query.
3. **Ask questions** related to the document.
4. The chatbot will **retrieve relevant context** and generate an answer.
5. View **chat history** for contextual follow-ups.

## Dependencies
- `streamlit`
- `langchain`
- `langchain_chroma`
- `langchain_community`
- `langchain_groq`
- `langchain_huggingface`
- `langchain_text_splitters`
- `python-dotenv`
- `chromadb`
- `pypdf`

## Future Improvements
- 📌 **Support for Multiple LLMs** (e.g., OpenAI, Anthropic)
- 📝 **Better UI/UX Enhancements** (e.g., Markdown support)
- 🔎 **Advanced Document Parsing** (e.g., tables & figures extraction)

🚀 **Happy Chatting with Your PDFs!**

