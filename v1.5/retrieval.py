from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_retriever():
    faiss_db_name = "faiss_index"
    bm25_db_name = "bm25_retriever.pkl"
    embeddings = GoogleGenerativeAIEmbeddings(
    	model="models/text-embedding-004",
    	google_api_key="AIzaSyC2TB9tiYPjJOUwmvAYk-DoceBprsdyDJM",
    	task_type="retrieval_query")

    faiss_db = FAISS.load_local(faiss_db_name, embeddings=embeddings,allow_dangerous_deserialization=True)
    faiss_retriever = faiss_db.as_retriever(search_type="mmr", search_kwargs={"k": 10,"fetch_k":20})

    with open(bm25_db_name, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                        weights=[0.4,0.6])
    return ensemble_retriever
