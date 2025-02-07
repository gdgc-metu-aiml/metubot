from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnableLambda

def get_retriever():
    faiss_db_name = "faiss_index"
    bm25_db_name = "bm25_retriever.pkl"
    model_path = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
                model_name = model_path,      
                model_kwargs = model_kwargs,  
                encode_kwargs = encode_kwargs 
            )

    faiss_db = FAISS.load_local(faiss_db_name, embeddings=embeddings,allow_dangerous_deserialization=True)
    faiss_retriever = faiss_db.as_retriever(search_type="mmr", search_kwargs={"k": 10,"fetch_k":30})

    with open(bm25_db_name, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                        weights=[0.4,0.6])
    
    limited_retriever = RunnableLambda(lambda query: ensemble_retriever.invoke(query)[:5])
    return limited_retriever