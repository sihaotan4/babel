
# Babel - Q&A bot with a document library

Babel is a simple Q&A chatbot that generates a response using relevant information from a document library. Users provide their own documents and ask questions about information in the library. Babel responds with knowledge from the library and avoids answering questions that are not related to the library. 

It is not intended for production use, but is a proof of concept to use a Large Language Model (LLM) with [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401).

_"Perhaps my old age and fearfulness deceive me, but I suspect that the human species -- the unique species -- is about to be extinguished, but the Library will endure: illuminated, solitary, infinite, perfectly motionless, equipped with precious volumes, useless, incorruptible, secret.”_
― Jorge Luis Borges, The Library of Babel

# Usage

Babel only has two actions:
1. Query - Ask a question and get an answer using information from the document library.
2. Reprocess documents - Re-process the documents to update the embeddings database if the content has changed.

```
BABEL IS STARTING...
READY - document and embeddings loaded into memory
SELECT ACTION: (1) Query, (2) Reprocess documents, (3) Quit: 1
Enter a question: what are some soup dishes?

ANSWER: Some soup dishes include beef noodle soup, banmian, sliced fish soup, fish soup bee hoon, soto ayam, soto, and cheng tng.

SOURCES (doc_name - para_id):
Beef noodle soup - 0ccd14f43e28405b955f03d798afac25
Banmian - 0b597d55af8b4090bcc33681983966d4
Sliced fish soup - f8e3d4c153534d66a4d84a664fa8a6c2
Fish soup bee hoon - c5033658552643dea00ef229bd2f4284
Beef noodle soup - 5f020efdeb93434b8599bee20e4857e4
Soto ayam - 45fef2a1395f4dd1b0ccbeb1fabd5b0a
Banmian - 5478c508945e4002bc186b023511d5e5
Soto (food) - c7ff33a2b1de4d2fb9d4ecf4a6114dde
Cheng tng - ccb567271f2e42d292ae52489552ea91
Beef noodle soup - 199dafe713204f4e82a81bb32f43e1bd
```
By default, Babel uses a toy dataset loaded in the `documents` folder. This dataset contains wikipedia pages about[ Singaporean cuisine](https://en.wikipedia.org/wiki/List_of_Singaporean_dishes). Therefore, Babel will only be able to answer questions related to Singaporean cuisine, specifically those that are described in the documents.

The user can provide easily load their own documents to create a document library specific to their use case. Once loaded, start Babel and use `(2) Reprocess documents` to generate the embeddings database for the documents. This only needs to be done once unless the content of the documents change.

# Setup

Pre-requisites:
- Python 3.9
- Your OpenAI API key (store this in `secrets/keys.json`)
- Your own documents in txt format (store these in `documents/`)

Steps:

Clone this repo.
Create a virtual environment and install requirements. If there are any issues with the installation, please update pip and try again.
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

## Document types and formats
The `read_documents()` function only supports simple txt files. This fucntion can be modified to support other file types like .doc, .pdf, .html, etc. 

## Chunking algorithm
The `fragment_doc()` function uses a simple chunking algorithm to split the documents into chunks with overlap. Depending on your document, it may be more appropriate to chunk the document by sections or clauses. This ensures that the chunk is properly self-contained and the embeddings generated are more relevant to the content of the chunk.

## Modifying chunked text
Chunks can be pre-processed to improve the quality of the embeddings generated. For example, you may want to remove the section headers or other text that is not relevant to the content of the chunk. This can be done in `enrich_text()`. A data sanitisation step can also be added to remove sensitive information if necessary. A chunk header can also be added to the chunk to provide more context before embedding generation (if for e.g. the subject is not mentioned in the chunk but is relevant to the context of the chunk).

## Embedding generation
The `get_embedding()` function can be modified to use a different model. There are embedding models that outperform openAI's "text-embedding-ada-002". These are highly dependent on the use case and the quality of the documents. Hosting an embedding model locally may be more cost effective if you are generating a large number of embeddings where as embedding via API calls may be more convenient for a smaller numbers of embeddings.

## Recovery from API errors
The `process_documents()` function has a single retry mechanism to recover from API errors. A better implementation would be to record the integrity of each row in the database and retry only the rows that failed.

## Vector similarity
The `nearest_neighbors()` function utilizes cosine similarity to find the most similar embeddings. It is a brute force approach that doesn't rely on any indexing. You may wish to use a specialized vector database that can implement a faster approximate nearest neighbor search. This is especially important if you are generating a large number of embeddings.

## Prompt engineering
The `engineer_prompt()` function contains the logic to fuse retrieved context to the original user_input.It also provides additional prompting to direct the LLM. Depending on the use case, you may wish to modify this function to improve the quality of the response.

# Disclaimer

### Safety and Security
Babel is not intended for production use. It uses outbound API calls to OpenAI's API on two occasions: 1) to generate the embeddings for the documents and 2) to generate the answer to the query. Be aware of the data security implications if you are using this tool with sensitive data.

### Costs
Babel uses OpenAI's API to generate the embeddings for the documents and to generate the answer to the query. The cost of these API calls are borne by the user. Please refer to OpenAI's pricing page for more information.

### Accuracy
The user is responsible for the accuracy of the documents and the questions. Babel does not perform any checks on the quality of the documents or the questions.
