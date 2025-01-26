from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
import pickle
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever():
    model_path = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
                model_name = model_path,      
                model_kwargs = model_kwargs,  
                encode_kwargs = encode_kwargs 
            )

    faiss_db = FAISS.load_local("faiss_index", embeddings=embeddings,allow_dangerous_deserialization=True)
    faiss_retriever = faiss_db.as_retriever(search_type="mmr", search_kwargs={"k": 3,"fetch_k":10})

    with open("bm25_retriever.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                        weights=[0.4,0.6])
    return ensemble_retriever
