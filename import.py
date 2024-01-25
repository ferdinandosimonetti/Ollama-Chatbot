from lang_funcs import *
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import os

# Loading phi from Ollama
llm = Ollama(model="neural-chat")
# Loading the Embedding Model
embed = load_embedding_model(model_path="all-MiniLM-L6-v2")

# loading and splitting the documents
path = 'documents' # Replace with your directory path here.

for filename in os.listdir(path):
    if filename.endswith('.pdf'):
        fullpath = path + os.sep + filename
        print(fullpath)
        # load doc
        docs = load_pdf_data(file_path=fullpath)
        documents = split_docs(documents=docs)
        # creating vectorstore
        vectorstore = create_embeddings(documents, embed)
        # converting vectorstore to a retriever
        retriever = vectorstore.as_retriever()

template = """
### System:
You are an respectful and honest assistant. You have to answer the user's questions using only the context \
provided to you. If you don't know the answer, just say you don't know. Don't try to make up an answer.

### Context:
{context}

### User:
{question}

### Response:
"""

# Creating the prompt from the template which we created before
prompt = PromptTemplate.from_template(template)

# Creating the chain
chain = load_qa_chain(retriever, llm, prompt)

# Asking questions
get_response("What's the best way to deploy an RDBMS instance on Kubernetes?",chain)
get_response("How can I expose a Deployment with a public IP address?",chain)

