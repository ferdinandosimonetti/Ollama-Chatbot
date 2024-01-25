from lang_funcs import *
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

import signal
import sys
def sigint_handler(signal, frame):
    print ("User has pressed CTRL+C / L'utente ha premuto CTRL+C")
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)

# Loading magiq-m0 from Ollama (with typo)
llm = Ollama(model="magicq-m0")
# Loading the Embedding Model
embed = load_embedding_model(model_path="all-MiniLM-L6-v2")

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
print("re-loading the embedding")
vectorstore = reload_embeddings(embedding_model=embed)
# converting vectorstore to a retriever
retriever = vectorstore.as_retriever()
print("creating the chain...")
# Creating the chain
chain = load_qa_chain(retriever, llm, prompt)
print("chain created.")

# Asking questions
#get_response("quali sono i sistemi usati per precomprimere il calcestruzzo?",chain)
#get_response("quali sono le norme italiane che regolano l'uso del calcestruzzo?",chain)
#get_response("quali sono le caratteristiche principali di un Cattivo di Fabula Ultima?",chain)
#get_response("What's the best way to deploy an RDBMS instance on Kubernetes?",chain)
#get_response("How can I expose a Deployment with a public IP address?",chain)
while True:
    try:
        question = input("Ask something to AI / Chiedi qualcosa all'AI ([INVIO] per uscire): ")
    except EOFError:
        print("Bye / Ciao")
        sys.exit(0)
    get_response(question,chain)
