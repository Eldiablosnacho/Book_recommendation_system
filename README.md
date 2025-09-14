This project is a Book Recommendation System that leverages HuggingFace embeddings, ChromaDB, and Gradio to deliver personalized book suggestions. It transforms book descriptions into vector embeddings for semantic similarity, stores them in a vector database, and allows users to interactively explore recommendations through a simple web interface.

Features:
  Data preprocessing and cleaning of book dataset
  Embedding generation using HuggingFace Inference API
  Vector storage and retrieval with ChromaDB
  Interactive search & filtering with Gradio UI
  Category-based dropdowns for refined recommendations

Tech Stack:
  Python
  Pandas, NumPy, Seaborn (data handling & analysis)
  HuggingFace Inference API (embeddings)
  ChromaDB (vector database)
  Gradio (UI for recommendations)

Project Workflow:
  Data Preparation → Load and clean the dataset.
  Embeddings → Convert book descriptions into vector representations.
  Database → Store embeddings in ChromaDB for fast retrieval.
  Search → Query the database for similar books.
  UI → Provide a Gradio-powered web app for user interaction.

How to Run:
  1. Clone the repository:
       git clone https://github.com/your-username/book-recommender.git
       cd book-recommender
  2. Install dependencies: pip install -r requirements.txt
  3. Set up environment variables in a .env file (for HuggingFace API key).
  4. Run the notebook or launch the Gradio app:
       python app.py


Developed as part of a learning project on Data Science, RAG, and Recommender Systems.
