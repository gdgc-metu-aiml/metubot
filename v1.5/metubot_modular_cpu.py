class ChatEngine():
    def __init__(self):
        import qa_chain_builder as cb
        import google_keys as gk
        from retrieval import get_retriever
        
        self.retriever = get_retriever()
        
        self.qa_chains = []
        for key in gk.google_keys:
            self.qa_chains.append(cb.qa_chain(key, self.retriever))

        self.keys = gk.google_keys
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
        
        result,retrieved_docs = self.qa_chains[self.current_qa_index].chat_with_qa(user_input.lower())
        self.update_api_key()
        return result,retrieved_docs
