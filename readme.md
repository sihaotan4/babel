
# Babel - Q&A from a set of documents

Babel is a simple CLI tool to answer questions using content from a document database. If there is no content in the documents that is relevant to the query, no response is generated (to limit the risk of hallucination).

It is not intended for production use, but is a proof of concept to test a Large Language Model (LLM) with Retrieval Augmented Generation (RAG).

_"Perhaps my old age and fearfulness deceive me, but I suspect that the human species -- the unique species -- is about to be extinguished, but the Library will endure: illuminated, solitary, infinite, perfectly motionless, equipped with precious volumes, useless, incorruptible, secret.”_
― Jorge Luis Borges, The Library of Babel

# Usage

Babel only has two actions:
1. Query - Ask a question and get an answer from the content of the documents.
2. Reprocess documents - Re-process the documents to update the embeddings if the content has changed.

# Setup

Pre-requisites:
- Python 3.9
- Your OpenAI API key (store this in `secrets/keys.json`)
- Your own documents* 
    
_*A toy dataset is provided in the `documents` folder. This dataset contains wikipedia pages about[ Singaporean cuisine](https://en.wikipedia.org/wiki/List_of_Singaporean_dishes). In terms of formatting, these should be simple txt files that can be stored in a nested folder structure._

Steps:

Create a virtual environment and install requirements. If there are any issues with the installation, please try updating pip and try again.
```
python -m venv venv_babel
venv_babel\Scripts\activate
pip install -r requirements.txt

# Update requirements.txt if you install new packages on your own
pip list --format=freeze > requirements.txt
```
Run Babel using the following command once you have navigated to the `babel` directory.
```
venv_babel\Scripts\python.exe main.py
```
# Technical Details

The following sections provide more details on the technical aspects of Babel and how it can be improved for your own use cases.

### Document types and formats
Babel only supports simple txt files. The `read_documents()` function in `main.py` can be modified to support other file types.

### Chunking algorithm
Babel uses a simple chunking algorithm in `fragment_doc()` to split the documents into chunks with overlap. Depending on your document, it may be more appropriate to chunk the document by sections or clauses. This ensures that the chunk is more self-contained and the embeddings generated are more relevant to the content of the chunk.

### Modifying chunked text
Chunks can be further modified to improve the quality of the embeddings generated. For example, you may want to remove the section headers or other text that is not relevant to the content of the chunk. This can be done in `fragment_doc()` or `enrich_text()`. A data sanitisation step can also be added to remove sensitive information if necessary. A chunk header can also be added to the chunk to provide more context before embedding generation (if for e.g. the subject is not mentioned in the chunk but is relevant to the context of the chunk).

### Embedding generation
The `get_embedding()` function in `main.py` can be modified to use a different model. There are embedding models that outperform openAI's "text-embedding-ada-002". These are highly dependent on the use case and the quality of the documents. Hosting a model locally may also be more cost effective if you are generating a large number of embeddings.

### Vector similarity
The `nearest_neighbors()` function utilizes cosine similarity to find the most similar embeddings. It is a brute force approach that doesn't rely on any indexing. You may wish to use a specialized vector database that can implement a faster approximate nearest neighbor search. This is especially important if you are generating a large number of embeddings.

### Prompt engineering
The `engineer_prompt()` function contains the logic to fuse retrieved context to the original user_input. Depending on the use case and the quality of the documents, you may wish to modify this function to improve the quality of the response.

### Content citation
By default, Babel doesn't return the source of the content. But this feature can be added by modifying the `get_answer()` function in `main.py`. The context is already retrieved as a full dataframe in `nearest_neighbors()` with all the necessary information to generate the citation.

# Disclaimer

### Safety and Security
Babel is not intended for production use. It uses outbound API calls to OpenAI's API on two occasions: 1) to generate the embeddings for the documents and 2) to generate the answer to the query. Be aware of the data security implications if you are using this tool with sensitive data.

### Costs
Babel uses OpenAI's API to generate the embeddings for the documents and to generate the answer to the query. The cost of these API calls are borne by the user. Please refer to OpenAI's pricing page for more information.

### Accuracy
The user is responsible for the accuracy of the documents and the questions. Babel does not perform any checks on the quality of the documents or the questions.