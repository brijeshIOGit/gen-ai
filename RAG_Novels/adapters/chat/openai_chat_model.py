class OpenAIChatModel:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def answer(self, question: str, context: str):
        prompt = f"""
        Use this context to answer the question.
        Context:
        {context}
        Question:
        {question}
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        usage = response.usage
        return {
            "text": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }