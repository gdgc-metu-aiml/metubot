from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

class qa_chain():
    def __init__(self, prompt, api_key, retriever):
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-4o-mini",
            temperature=0.3
        )
        
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False)
    
    def chat_with_qa(self, user_input):
        result = self.qa.invoke(user_input)
        return result["result"]
