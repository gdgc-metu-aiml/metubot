import os
import pandas as pd
import pickle
import faiss
from langchain_community.document_loaders import CSVLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

model_path = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
            model_name = model_path,      
            model_kwargs = model_kwargs,  
            encode_kwargs = encode_kwargs 
        )

dimension = len(embeddings.embed_query("hi"))
empty_index = faiss.IndexFlatL2(dimension)
docstore = InMemoryDocstore({})
faiss_db = FAISS(embedding_function=embeddings, index=empty_index,index_to_docstore_id={},docstore=docstore)

folder = './data/'
files = os.listdir(folder)

docs= []
for file in files:
    # CSV Loader
    if file.endswith('.csv'):
        df = pd.read_csv(folder + file)
        source = df.columns[-1]
        docs.extend(CSVLoader(file_path=folder + file,csv_args={'delimiter': ','},encoding='utf-8',content_columns=[source]).load())
    # PDF Loader
    elif file.endswith('.pdf'):
        docs.extend(PyPDFLoader(folder + file).load())
    print(file)
print(len(docs))

#Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,)
splitted_docs = text_splitter.split_documents(docs)
print(len(splitted_docs))

for id,doc in enumerate(splitted_docs):
    doc.id = str(id)
    doc.page_content = doc.page_content.lower()

for id,doc in enumerate(splitted_docs):
    doc.id = id
    doc.page_content = doc.page_content.lower()

faiss_db.add_documents(splitted_docs)
faiss_db.save_local("faiss_index_semantic")

bm25_retriever = BM25Retriever.from_documents(documents=splitted_docs)
with open("bm25_retriever_semantic.pkl", "wb") as f:
    pickle.dump(bm25_retriever, f)

tokenizer = AutoTokenizer.from_pretrained("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
for i in splitted_docs:
    tokens = tokenizer.tokenize(i.page_content)
    if len(tokens)>512:
        print(len(tokens))

print("Done")