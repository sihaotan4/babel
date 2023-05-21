import pandas as pd
import numpy as np
import glob
import os
import json
import uuid
import threading
import openai
import tiktoken
from blingfire import text_to_sentences
from time import sleep
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
tqdm.pandas()
import logging

# initialize global variables and logging
PATH_TO_DATABASE = os.getcwd() + '/documents'
CHUNK_SIZE = 5 # number of sentences per paragraph
CHUNK_OVERLAP = 1 # number of sentences to overlap between paragraphs
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_TOKEN_LIMIT = 8191-2 # account for the 2 tokens added by the model
EMBEDDING_LENGTH = 1536 # length of the embedding vector
EMBEDDING_BATCH_SIZE = 25 # number of paras to embed at once

ACTIVE_MODEL = "gpt-3.5-turbo"
ACTIVE_TOKEN_LIMIT = 4096-2

logging.basicConfig(filename=f'{os.getcwd()}/logs/babel.log', 
                                format='%(asctime)s-%(levelname)s-%(message)s', 
                                level=logging.INFO)
with open(os.getcwd() + '/secrets/keys.json') as f:
    keys = json.load(f)
openai.api_key = keys['OPENAI_API_KEY']
enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)

### FUNCTIONS ###

def get_paths():
    """
    Get paths to all documents in the database. Ignore paths which are directories
    Return: list of absolute file paths
    """
    doc_paths = [f for f in glob.glob(PATH_TO_DATABASE + '/**', recursive=True) if not os.path.isdir(f)]
    return [i.replace('\\', '/') for i in doc_paths]

def fragment_doc(doc):
    """
    Split a document into paragraphs of sentences.
    Returns a dict with info about the document, including the list of paragraphs.
    """
    paragraphs = []
    file_token_count = 0
    # use blingfire sentence boundary detection to split document into sentences
    sents = text_to_sentences(doc).split('\n')
    # combine sentences into paragraphs with overlap
    for i in range(0, len(sents), CHUNK_SIZE-CHUNK_OVERLAP):
        # using a slice, so even if [i:i+CHUNK_SIZE] is out of bounds, 
        # it will just return the remaining sentences
        # this is useful for the last paragraph, which may not have CHUNK_SIZE sentences
        para = ' '.join(sents[i:i+CHUNK_SIZE])
        # check if paragraph is too long for the embedding model
        file_num_tokens = len(enc.encode(para))
        if file_num_tokens > EMBEDDING_TOKEN_LIMIT:
            raise RuntimeError(f'{doc_name}: Paragraph too long: {para[:50]}...')
        # paragraph is short enough, so just add it to the list
        else:
            paragraphs.append(para)
        # update token count
        file_token_count += file_num_tokens
    # store info about the document in a dict
    doc_data = {'num_sents': len(sents),
                'num_paras': len(paragraphs),
                'num_tokens': file_token_count,
                'text': paragraphs}
    return doc_data

def read_documents():
    """
    Read all documents in the database into a list of dicts.
    Each dict contains info about the document, including the list of paragraphs.
    """
    doc_paths = get_paths()
    docs = []
    for doc_path in doc_paths:
        # read document
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc = f.read()
        doc_name = doc_path.split('/')[-1]
        # split document into paragraphs
        doc_data = fragment_doc(doc)
        # add document name and path to dict
        doc_data['doc_name'] = doc_name
        doc_data['doc_path'] = doc_path
        docs.append(doc_data)
    return docs

def calculate_cost(document_df):
    """Takes the prepared dataframe of documents and calculates metadata for entire database."""
    costs = {'num_docs': len(document_df),
                 'num_paras': document_df['num_paras'].sum(),
                 'num_tokens': document_df['num_tokens'].sum(),
                 'est_cost_usd': np.round((document_df['num_tokens'].sum() * 0.0004 / 1000), 2)}
    return costs

def format_for_embedding(df):
    """
    Explode the paragraphs into separate rows, generate a unique id for each paragraph.
    Drop columns that are no longer needed.
    Return: dataframe with one paragraph per row, and a unique id for each paragraph
    """
    df = df.copy()
    # explode the paragraphs into separate rows, generate a unique id for each paragraph
    df = df.explode('text').reset_index(drop=True)
    df = df.drop(columns=['num_sents', 'num_paras', 'num_tokens'])
    df['id']= [uuid.uuid4().hex for _ in range(len(df))]
    while len(df['id'].unique()) != len(df):
        df['id']= [uuid.uuid4().hex for _ in range(len(df))]
    # calculate tokens for each paragraph
    df['num_tokens'] = df['text'].apply(lambda x: len(enc.encode(x)))
    return df

def enrich_texts(df, col_name):
    df = df.copy()
    df['enriched_text'] = df['doc_name'] + ': ' + df[col_name]
    return df

def get_embedding(text):
    text = text.replace("\n", " ")
    try:
        res = openai.Embedding.create(input = [text], model=EMBEDDING_MODEL)['data'][0]['embedding']
    except openai.error.OpenAIError as e:
        # retry once
        sleep(5)
        try:
            res = openai.Embedding.create(input = [text], model=EMBEDDING_MODEL)['data'][0]['embedding']
        except openai.error.OpenAIError as e:
            res = np.zeros(EMBEDDING_LENGTH)
            print("ERROR: Unable to embed text:", text)
            logging.error(f'Unable to embed text: {text}. Error: {json.dumps(e)}')
    return res

def embed_documents(df, col_name):
    df = df.copy()
    df['embedding'] = df[col_name].progress_apply(get_embedding)
    return df

def process_documents():
    """
    Function to ingest, embed, and store document data and embeddings.
    """
    logging.info('process_documents(1/4): Attempting to process document database')
    docs = read_documents()
    df = pd.DataFrame(docs)
    costs = calculate_cost(df)
    print('documents loaded into memory')
    print(f'details: {costs}')
    logging.info(f'process_documents(2/4): Documents loaded into memory. Details: {costs}')

    print('embedding documents...')
    df = format_for_embedding(df)
    df = enrich_texts(df, 'text')
    df = embed_documents(df, 'enriched_text') # Note: this embeds the enriched text
    logging.info('process_documents(3/4): Embedding complete')
    # store df as pickle in data directory
    df.to_pickle(os.getcwd() + '/data/embeddings.pkl')
    print('embeddings dataframe saved to data/embeddings.pkl')
    logging.info('process_documents(4/4): Successfully processed document database, data saved to data/embeddings.pkl')
    return None

def nearest_neighbors(query_embedding, df, col_name, token_limit=1500, similarity_threshold=0.75, k=10):
    """
    Find the nearest neighbors for a given embedding vector.
    Uses cosine similarity to find the nearest neighbors.
    Returns: dataframe with the nearest neighbors filtered by token limit, similarity threshold, and k
    """
    df = df.copy()
    # calculate cosine similarity between query embedding and all embeddings in df
    df['similarity'] = df[col_name].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
    df = df.sort_values(by='similarity', ascending=False)
    # apply constraints
    constraints = {'k': k, 
                   'token_limit': sum(df.num_tokens.cumsum() < token_limit),
                   'similarity_threshold': sum(df.similarity > similarity_threshold)}
    # find the smallest constraint
    constraint = min(constraints, key=constraints.get)
    # slice df to include rows limited by the smallest constraint
    df = df.iloc[:constraints[constraint]]      
    
    return df, constraint

def engineer_prompt(prompt, context=None):
    """Converts a prompt into openai's message list format."""

    query =  f"""Use the below \"\"\"documents\"\"\" to answer the subsequent question. If \"\"\"documents\"\"\" is not relevant to the QUESTION, reply with only three words: "I don't know".

\"\"\"documents\"\"\":
\"\"\"
{context}
\"\"\"

QUESTION: {prompt}"""

    messages = [{"role": "system", "content": "You answer questions using the \"\"\"documents\"\"\"."},
                {"role": "user", "content": query}]
    return messages

def get_answer(input, df):
    """Takes a user input and returns the answer from the knowledge base."""
    logging.info(f'get_answer(1/3): received input: {input}')
    input_embedding = get_embedding(input)
    results, constraint = nearest_neighbors(input_embedding, df, 'embedding')
    # if results dataframe is empty, return None
    if results.empty:
        logging.info(f'get_answer(2/3): No results due to constraint:{constraint}')
        print(f'No results found. Please try again with a different input.')
        return None, constraint
    else:
        context = results['enriched_text'].str.cat(sep='\n')
        logging.info(f'get_answer(2/3): context retrieved:{context}')
    # get answer from model with engineered prompt
    messages = engineer_prompt(input, context)
    try:
        answer = openai.ChatCompletion.create(model=ACTIVE_MODEL, messages=messages, temperature=0)
    except openai.error.OpenAIError as e:
        error_code = e.response['error']['code']
        print(f'Please try again later. Error code: {error_code} from OpenAI')
        logging.info(f'get_answer(3/3): Unsuccessful. Error code: {error_code} from OpenAI.ChatCompletion.create()')
        return None, constraint
    logging.info(f'get_answer(3/3): Successful. Details: {json.dumps(answer)}')
    return answer, constraint

def print_spinner():
    global spinning
    spinning = True
    t = threading.Thread(target=spin)
    t.start()

def spin():
    while spinning:
        for char in '|/-\\':
            print(f"\r{char} ", end='')
            sleep(0.1)

def stop_spinner():
    global spinning
    spinning = False

### MAIN ###

def main():
    print('BABEL IS STARTING...')
    # load documents and embeddings into memory
    if os.path.exists(os.getcwd() + '/data/embeddings.pkl'):
        try: 
            df = pd.read_pickle(os.getcwd() + '/data/embeddings.pkl')
            print('READY - document and embeddings loaded into memory')
        except (pickle.UnpicklingError, FileNotFoundError) as e:
            print('ERROR: Unable to load embeddings.pkl')
            logging.error(f'Unable to load embeddings.pkl. Error: {str(e)}')
    
    # create a loop to continue asking for user input (query, reprocess, or quit)
    while True:
        user_input = input('SELECT ACTION: (1) Query, (2) Reprocess documents, (3) Quit: ')
        if user_input == '1':
            user_input = input('Enter a question: ')
            print_spinner()
            answer, _ = get_answer(user_input, df)
            stop_spinner()
            sleep(0.2)
            if answer:
                print(f"\n{answer.choices[0]['message']['content']}")
        elif user_input == '2':
            costs = calculate_cost(pd.DataFrame(read_documents()))
            print(f'Reprocess documents - reprocessing all documents would cost: {costs["est_cost_usd"]} USD')
            user_input = input('Would you like to continue? (y/n): ')
            while user_input not in ['y', 'n']:
                user_input = input('Please enter "y" or "n": ')
            if user_input == 'y':
                process_documents()
                df = pd.read_pickle(os.getcwd() + '/data/embeddings.pkl')
                print('READY - document and embeddings loaded into memory')
        elif user_input == '3':
            print('BYE...')
            break
        else:
            print('Please select a valid action (1, 2, or 3)')

if __name__ == '__main__':
    main()