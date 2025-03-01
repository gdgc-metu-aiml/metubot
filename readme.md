# metubot - AI Assistant Chatbot for University Students

Official repository of **Metubot AI Assistant Chatbot Architecture**\
**Official implementation:** Hector - AI Assistant Chatbot for METU (Middle East Technical University) Students

## 📌 Overview

Metubot is an AI-powered chatbot architecture specifically designed for university students, with a primary focus on assisting university students in Turkish. It utilizes a **Retrieval-Augmented Generation (RAG)** architecture with **Faiss-based vector search, BM25 retrieval, and OpenAI's LLMs** to deliver accurate and context-aware responses tailored to academic and university-related queries.

## 🚀 Features

- 🔍 **Hybrid Search:** Faiss for dense vector retrieval and BM25 for keyword-based search.
- 🌍 **Multilingual Support:** Designed to handle both Turkish and English queries efficiently.
- 📚 **Knowledge Categorization:** Organized data retrieval for improved accuracy.
- 🤖 **LLM Integration:** Uses OpenAI models for response generation.
- 🛠 **Customizable & Extensible:** Easily adaptable to different datasets and retrieval mechanisms.

## 🔧 Installation

To set up the Metubot project on your local machine:

```bash
git clone https://github.com/gdgc-metu-aiml/metubot.git
cd metubot
pip install -r requirements.txt
```

## 📂 Required Files & Keys

Users must generate their own database files. Place the required database and API key files in the `v1.6` directory as follows:

```
v1.6/faiss_index/
v1.6/bm25_retriever.pkl
v1.6/openai_keys.py
```

- `faiss_index/`: Directory containing the Faiss index files.
- `bm25_retriever.pkl`: Precomputed BM25 retriever model.
- `openai_keys.py`: File containing API keys for OpenAI integration.

## 💬 Usage

To use Metubot in different applications, import `v1.6/metubot_modular_cpu.py` and create an instance of the `ChatEngine()` class. You can interact with the chatbot using:

```python
from metubot_modular_cpu import ChatEngine
chatbot = ChatEngine()
response, docs = chatbot.chat("Your prompt here")
print(response)
```

For terminal-based interaction:

```bash
python3 ./v1.6/chat.py
```

## 📜 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for more details.

## 👥 Contributors

Metubot is developed by members of the **Google Developer Groups on Campus METU AI/ML Team**.

- **Project Lead:** Efe Kaan Güler

- **Architecture Development Team:** Efe Kaan Güler, Özgür Yıldırım

- **Data Team:** Özgür Yıldırım, Yavuz Alp Demirci, Begüm Atay

- **Backend Development Team:** Emre Ekiz

- **Mobile/Web App Development Team:** Eray Kaya

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

For inquiries or technical questions, contact **Efe Kaan Güler** at **[efekaanguler05@gmail.com](mailto\:efekaanguler05@gmail.com)** or open an issue in this repository.


