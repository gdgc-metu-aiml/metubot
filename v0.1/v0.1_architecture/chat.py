from metubot_modular_cpu import ChatEngine

def main():
    chat_engine = ChatEngine()
    
    print("Chatbot ile sohbet etmeye hoş geldiniz! Çıkmak için 'exit' yazın.")
    
    while True:
        user_input = input("Siz: ")
        if user_input.lower() == 'exit':
            print("Sohbet sona erdi.")
            break
        
        response = chat_engine.chat(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()