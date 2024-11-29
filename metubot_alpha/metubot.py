import numpy as np
import pandas as pd
import torch
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline
from transformers import BertForQuestionAnswering, BertTokenizer
from transformers import AutoModelForCausalLM, AutoModel
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

csv_file_path = 'example_data.csv'

loader = CSVLoader(file_path=csv_file_path)

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

docs = text_splitter.split_documents(data)

model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name = model_path,      
    model_kwargs = model_kwargs,  
    encode_kwargs = encode_kwargs 
)

text = 'Üreten ekibe hoş geldin'
query_result = embeddings.embed_query(text)
len(query_result)

vector_db = FAISS.from_documents(docs, embeddings)

retriever = vector_db.as_retriever()


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.1,
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)
print("\n\n----------------------------------------------\n\n")
print("metubot:   Merhaba, user sana nasıl yardımcı olabilirim?")
while True:
	question = input("user:      ")
	start_time = datetime.now()
	result = qa.invoke(question)
	end_time = datetime.now()
	print("metubot:   "+result['result'])
	elapsed_time = end_time - start_time
	#print(f"Cevap süresi: {elapsed_time}")
	
	



