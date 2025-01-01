class ChatEngine():
    def __init__(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import CSVLoader
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain_groq import ChatGroq
        import groq_keys as gk
        from retrieval import get_retriever

        self.current_api_key = gk.groq_api_key
        
        self.retriever = get_retriever()
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            api_key=self.current_api_key)
        
        self.template = """
        Sen, ODTÜ öğrencilerine akademik, sosyal ve idari konularda yardımcı olmak için tasarlanmış bir yapay zeka asistanısın. Görevin, yalnızca sana verilen dokümanlarda bulunan bilgilere dayanarak doğru ve bağlamlı cevaplar vermektir.
        Eğer sorulan soru dokümanlarda bulunmuyorsa ya da yanıt için yeterli bilgi yoksa, bunu açıkça belirt ve 'Bu konuda elimde bilgi bulunmamaktadır.' de.
        Asla tahmin yapma veya dokümanda olmayan bir bilgi üretme.
        Aksi belirtilmedikçe Türkçe cevap ver.
        Yanıtların kısa, net ve kullanıcıya faydalı olmalıdır.
        Sorunun bağlamını tam anlamazsan, daha fazla detay isteyebilirsin.
        Şimdi, kullanıcıların sorularına verilen dokümanlardaki bilgilere dayanarak yanıt ver.
        {context}
        Öğrenci: {question}
        Sen:"""

        self.prompt = PromptTemplate(
            template = self.template,
            input_variables = ["question", "context"])

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=False)

    def chat(self, user_input):
        """
        This function takes user input as string and returns the response from the model as string.
                
        Args: 
        user_input (str): User input as string.
        """
                
        result = self.qa.invoke(user_input)
        return result["result"]
