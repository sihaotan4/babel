
# Babel

A simple project to test a Large Language Model (LLM) with Retrieval Augmented Generation (RAG). 

Takes a user query and generates a response using content from a set of documents. If there is no content in the documents that is relevant to the query, no response is generated (to limit the risk of hallucination).

_"Perhaps my old age and fearfulness deceive me, but I suspect that the human species -- the unique species -- is about to be extinguished, but the Library will endure: illuminated, solitary, infinite, perfectly motionless, equipped with precious volumes, useless, incorruptible, secret.”_
― Jorge Luis Borges, The Library of Babel



## Workflow
1. Point to a folder with a set of documents. Ingest and chunk documents into paragraphs with overlap.
2. Embed paragraphs into a vector space. 
3. Insert vectors into a Pinecone index (or really any vector database).
4. Take a query and return vectors that are most similar to the query. 
5. Map vectors back to paragraphs and returns paragraphs.
6. Assemble paragraphs into context.
7. Generate response using context and query.

## Setup

Create a virtual environment and install requirements.
```
python -m venv venv_babel
venv_babel\Scripts\activate
pip install -r requirements.txt
```
Update requirements.txt if you install new packages.
```
pip list --format=freeze > requirements.txt
```