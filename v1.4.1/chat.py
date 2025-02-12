from metubot_modular_cpu import ChatEngine
import datetime

def main():
    chat_engine = ChatEngine()
    
    print("Chatbot ile sohbet etmeye hoş geldiniz! Çıkmak için 'exit' yazın.")
    
    while True:
        user_input = input("Siz: ")
        if user_input.lower() == 'exit':
            print("Sohbet sona erdi.")
            break
        
        start = datetime.datetime.now()

        response,docs = chat_engine.chat(user_input)
        
        stop = datetime.datetime.now()
        print(f"Chatbot: {response} \n {docs}")
        print(f"Time: {stop-start} sec.")

if __name__ == "__main__":
    main()
