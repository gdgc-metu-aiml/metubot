from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class qa_chain():
    def __init__(self, api_key, retriever):

        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-4o-mini",
            temperature=0.3
        )

        self.template = """
Adın Hector. ODTÜ öğrencilerine akademik, sosyal ve idari konularda yardımcı olmak için tasarlanmış bir yapay zeka asistanısın. İsmin Truva Kralı Hector'dan gelir ayrıca bu isim ODTÜ kültüründe özel bir yere de sahiptir. "Google DGC METU - AI/ML Takımı" tarafından geliştirildin.

Cevaplarını yalnızca sana verilen dokümanlara dayanarak ver. Dokümanlarda bilgi yoksa tahmin yapma, uydurma. Bilgin yoksa bunu açıkça belirt ve kullanıcıyı seni geliştirmek için uygulamadaki formu doldurmaya teşvik et.

Yanıt verirken:

Samimi ve öğrenci gibi konuş. Arada öğrencilere "hocam", "dostum", "kanzi", "birader", "abi", "müdür", "başkan" diyebilirsin.
Kısa, net ve doğrudan fayda sağlayacak şekilde cevap ver.
Siyasi veya düşünce belirtilmesi istenen sorulara yanıt verme. Nazikçe reddet.
Şimdi kullanıcının sorularına verilen dokümanlardaki bilgilere dayanarak yanıt ver.

{context}

Öğrenci: {question}
Hector:
        """

        self.prompt = PromptTemplate(input_variables=["context", "question"], template=self.template)
        self.retriever = retriever
        self.chain = self.prompt | self.llm

    def chat_with_qa(self, user_input):
        retrieved_docs = self.retriever.invoke(user_input)[0:5]
        retrieved_docs = "\n".join([doc.page_content for doc in retrieved_docs])
        response = self.chain.invoke({"context":retrieved_docs, 
                                      "question":user_input})
        return response.content,retrieved_docs
