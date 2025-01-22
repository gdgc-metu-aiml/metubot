from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

class qa_chain():
    def __init__(self, prompt, api_key, retriever):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            api_key=api_key)
        
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False)
    
    def chat_with_qa(self, user_input):
        result = self.qa.invoke(user_input)
        return result["result"]