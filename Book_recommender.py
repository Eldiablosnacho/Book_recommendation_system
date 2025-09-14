#downloading the data
from dataset import path

print("Path to dataset files:", path)

import pandas as pd

books = pd.read_csv(f"{path}/books.csv")
books.head()

import seaborn as sns
import matplotlib.pyplot as plt

#cleaning the data
ax = plt.axes()
sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)

plt.xlabel("Column")
plt.ylabel("Missing values")
plt.show()


import numpy as np

books["missing_description"] = np.where(books["description"].isna(), 1, 0)
books["age_of_book"] = 2025 - books["published_year"]

column_of_interest = ["num_pages", "age_of_book", "missing_description", "average_rating"]

correlation_matrix = books[column_of_interest].corr(method = 'spearman')


sns.set_theme(style="white")
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label":"Spearman correlation"})
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.show()

book_missing = books[~(books["description"].isna())|
      ~(books["num_pages"].isna()) |
      ~(books["average_rating"].isna()) |
      ~(books["published_year"].isna())]

book_missing

book_missing["categories"].value_counts().reset_index().sort_values(by="count", ascending = False)

book_missing["words_in_description"] = book_missing["description"].str.split().str.len()
book_missing

book_missing.loc[book_missing["words_in_description"].between(1, 4), "description"]

book_missing.loc[book_missing["words_in_description"].between(5, 14), "description"]

book_missing.loc[book_missing["words_in_description"].between(15, 24), "description"]

book_missing.loc[book_missing["words_in_description"].between(25, 34), "description"]

book_missing_25_words = book_missing[book_missing["words_in_description"] >=25]
book_missing_25_words

book_missing_25_words["title_and_subtitle"] = (
    np.where(book_missing_25_words["subtitle"].isna(), book_missing_25_words["title"],
             book_missing_25_words[["title", "subtitle"]].astype(str).agg(":".join, axis=1))
)

book_missing_25_words["tagged_description"] = book_missing_25_words[["isbn13", "description"]].astype(str).agg(":".join, axis=1)
book_missing_25_words

(
    book_missing_25_words
    .drop(["subtitle", "missing_description", "age_of_book", "words_in_description"], axis=1)
    .to_csv("books_cleaned.csv", index = False)
)

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from API_keys import embeddings

import pandas as pd

books = pd.read_csv("books_cleaned.csv")
books.head()

books["tagged_description"]

books["tagged_description"].to_csv("tagged_description.txt", sep="\n", index=False, header=False)

raw_documents = TextLoader("tagged_description.txt").load()
# Each line in the text file is already a separate document
text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n") #chunk_size can't be zero
documents = text_splitter.create_documents([raw_documents[0].page_content])

documents[0]

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import gradio as gr
import os

#  Loading the dataset
df = pd.read_csv("books_cleaned.csv")

#  Prepare documents for Chroma
documents = []
for i, row in df.iterrows():
    text = f"Title: {row['title']}\nAuthor: {row['authors']}\nCategory: {row['categories']}\nDescription: {row['tagged_description']}"
    metadata = {
        "title": row['title'],
        "author": row['authors'],
        "category": row['categories'],
        "rating": row['average_rating'],
        "thumbnail": row['thumbnail'],
        "year": row['published_year']
    }
    documents.append(Document(page_content=text, metadata=metadata))


#  Create HuggingFace Embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#  Create or Load Chroma DB (Persistent)
persist_dir = "books_chroma_db"
if os.path.exists(persist_dir):
    db_books = Chroma(persist_directory=persist_dir, embedding_function=embedding)
else:
    db_books = Chroma.from_documents(documents, embedding=embedding, persist_directory=persist_dir)
    db_books.persist()

#  Gradio search function
def recommend_books(query, min_rating, category):
    filters = {}
    if category != "All":
        filters["category"] = category
    if min_rating > 0:
        filters["rating"] = {"$gte": min_rating}

    results = db_books.similarity_search(query, k=5)

    recommendations = []
    for res in results:
        meta = res.metadata
        recommendations.append(f"📖 **{meta['title']}** by {meta['author']} ({meta['year']})\n"
                               f"⭐ Rating: {meta['rating']}\n"
                               f"Category: {meta['category']}\n"
                               f"![]({meta['thumbnail']})\n")
    return "\n\n".join(recommendations)



#  Get unique categories for dropdown
categories = ["All"] + sorted(set(df['categories'].dropna().unique()))


#  Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📚 Book Recommender System")
    query = gr.Textbox(label="Enter your interest or book title")
    category = gr.Dropdown(choices=categories, value="All", label="Select Category")
    min_rating = gr.Slider(0, 5, 0, step=0.5, label="Minimum Rating")
    output = gr.Markdown()
    btn = gr.Button("Recommend Books")

    btn.click(fn=recommend_books, inputs=[query, min_rating, category], outputs=output)

demo.launch()