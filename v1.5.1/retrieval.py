from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import google_keys as gk

def get_retriever():
    faiss_db_name = "google_faiss_index"
    google_embeddings = GoogleGenerativeAIEmbeddings(
    	model="models/text-embedding-004",
    	google_api_key=gk.google_keys[0],
    	task_type="retrieval_query")

    google_faiss_db = FAISS.load_local(faiss_db_name, embeddings=google_embeddings,allow_dangerous_deserialization=True)
    google_faiss_retriever = google_faiss_db.as_retriever(search_type="mmr", search_kwargs={"k": 10,"fetch_k":20})
    
        
    faiss_db_name = "emrecan_faiss_index"
    model_path = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    emrecan_embeddings = HuggingFaceEmbeddings(
                model_name = model_path,      
                model_kwargs = model_kwargs,  
                encode_kwargs = encode_kwargs)

    emrecan_faiss_db = FAISS.load_local(faiss_db_name, embeddings=emrecan_embeddings,allow_dangerous_deserialization=True)
    emrecan_faiss_retriever = emrecan_faiss_db.as_retriever(search_type="mmr", search_kwargs={"k": 10,"fetch_k":20})
    
    bm25_db_name = "bm25_retriever.pkl"
    with open(bm25_db_name, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, google_faiss_retriever, emrecan_faiss_retriever],
                                        weights=[1,1,1])
    return ensemble_retriever
