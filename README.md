# News Research Tool 📰

A Streamlit-based web application that allows you to research and ask questions about news articles using AI. The tool uses Google's Gemini AI and FAISS vector store to provide intelligent answers with source attribution.

## Features

- Load multiple news article URLs (up to 3 at a time)
- Automatically extracts and processes article content
- Creates embeddings and stores them in a FAISS vector database
- Ask questions about the articles and get AI-generated answers
- Saves the FAISS index locally for reuse without re-processing URLs

## Prerequisites

- Python 3.8 or higher
- Google API key (for Gemini AI)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd news-research-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. In the sidebar, enter up to 3 news article URLs

3. Click "Process URLs" to load and process the articles

4. Enter your research question in the text input field

5. Click "Ask" to get an AI-generated answer based on the articles

## How It Works

1. **Document Loading**: Articles are loaded from the provided URLs using WebBaseLoader
2. **Text Splitting**: Documents are split into manageable chunks for better processing
3. **Embeddings**: Text chunks are converted to embeddings using Google's Gemini embedding model
4. **Vector Storage**: Embeddings are stored in a FAISS vector database for efficient retrieval
5. **Question Answering**: When you ask a question, the most relevant chunks are retrieved and sent to Gemini AI to generate an answer

## Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: Framework for building LLM applications
- **Google Gemini AI**: Language model for embeddings and question answering
- **FAISS**: Vector database for efficient similarity search
- **Python-dotenv**: Environment variable management

## License

MIT License

## Contributing

Feel free to submit issues and enhancement requests!
