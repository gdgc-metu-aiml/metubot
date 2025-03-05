class ChatEngine():
    def __init__(self):
        import qa_chain_builder as cb
        import openai_keys as gk
        from retrieval import get_retriever
        
        self.retriever = get_retriever()
        
        self.qa_chains = []
        for key in gk.openai_keys:
            self.qa_chains.append(cb.qa_chain(key, self.retriever))

        self.keys = gk.openai_keys
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

    def chat(self, user_input, user_name):
        """
        This function takes user input and user name as string and returns:
        result, retrieved_docs
                
        Args: 
        user_input (str): User input as string.
        user_name (str): User name as string.
        """
        if user_name == "Misafir":
            name_definer = "(İsim belirtilmedi) "
        else:
            name_definer = f"(İsim: {user_name}) "
        processed_input = name_definer + user_input.lower()
        
        result,retrieved_docs = self.qa_chains[self.current_qa_index].chat_with_qa(processed_input)
        self.update_api_key()
        return result,retrieved_docs
