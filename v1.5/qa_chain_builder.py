from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

class qa_chain():
    def __init__(self, api_key, retriever):

        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model="gemini-2.0-flash",
            temperature=0.3
        )

        self.template = """\
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

        self.prompt = PromptTemplate(input_variables=["context", "question"], template=self.template)
        self.retriever = retriever
        self.chain = self.prompt | self.llm

    def chat_with_qa(self, user_input):
        retrieved_docs = self.retriever.invoke(user_input)[0:5]
        retrieved_docs = "\n".join([doc.page_content for doc in retrieved_docs])
        response = self.chain.invoke({"context":retrieved_docs, 
                                      "question":user_input})
        return response.content,retrieved_docs
