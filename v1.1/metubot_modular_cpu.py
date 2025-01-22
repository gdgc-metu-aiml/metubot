class ChatEngine():
    def __init__(self):
        from langchain.prompts import PromptTemplate
        import qa_chain_builder as cb
        import groq_keys as gk
        from retrieval import get_retriever
        
        self.retriever = get_retriever()
        
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
        
        self.qa_chains = []
        for key in gk.groq_api_keys:
            self.qa_chains.append(cb.qa_chain(self.prompt, key, self.retriever))

        self.keys = gk.groq_api_keys
        self.num_keys = len(self.keys)
        self.current_qa_index = 0
        self.current_api_key = self.keys[self.current_qa_index]

    def update_api_key(self):
        if self.current_qa_index == self.num_keys - 1:
            self.current_qa_index = 0
            self.current_api_key = self.keys[self.current_qa_index]
        else: 
            self.current_qa_index += 1
            self.current_api_key = self.keys[self.current_qa_index]

    def chat(self, user_input):
        """
        This function takes user input as string and returns the response from the model as string.
                
        Args: 
        user_input (str): User input as string.
        """
        
        result = self.qa_chains[self.current_qa_index].chat_with_qa(user_input)
        self.update_api_key()
        return result