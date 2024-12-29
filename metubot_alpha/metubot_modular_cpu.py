class ChatEngine():
    def __init__(self):
        import numpy as np
        import pandas as pd
        from datetime import datetime
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import CSVLoader
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.chains import RetrievalQA
        from langchain_ollama import ChatOllama
        
        self.csv_file_path = 'example_data.csv'
        self.loader = CSVLoader(file_path=self.csv_file_path)

        self.data = self.loader.load()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        
        self.docs = self.text_splitter.split_documents(self.data)

        self.model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'normalize_embeddings': False}

        self.embeddings = HuggingFaceEmbeddings(
            model_name = self.model_path,      
            model_kwargs = self.model_kwargs,  
            encode_kwargs = self.encode_kwargs 
        )

        self.vector_db = FAISS.from_documents(self.docs, self.embeddings)
        self.retriever = self.vector_db.as_retriever()

        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0.1,
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever)

    def chat(self, user_input):
        """
        This function takes user input as string and returns the response from the model as string.
                
        Args: 
        user_input (str): User input as string.
        """
                
        result = self.qa.invoke(user_input)
        return result['result']
