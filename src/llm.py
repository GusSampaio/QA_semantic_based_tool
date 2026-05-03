from langchain_mistralai import ChatMistralAI
from src.settings import AppSettings, missing_required_env

class LLM:
    def __init__(self, settings: AppSettings):
        self.model = self.build_llm(settings)
    
    def build_llm(self, settings: AppSettings) -> ChatMistralAI:
        if missing_required_env(settings):
            raise ValueError("MISTRAL_API_KEY não encontrada. Verifique o arquivo .env na raiz do projeto.")

        return ChatMistralAI(
            model=settings.chat_model,
            api_key=settings.mistral_api_key,
            temperature=0.2,
        )

    def generate_answer(self, question: str, context: str) -> str:
        prompt =  f"""Você é um assistente de tarefas de pergunta e resposta. Use as partes seguintes do contexto recuperado por um grafo (representada por tuplas) para responder a pergunta. O contexto considera papéis semânticos anotados
Pergunta: {question}
Contexto: {context}"""
        return prompt
    

    def answer_question_with_llm(
        self,
        question: str,
        tuplas_grafo: str,
    ) -> str:
        """Gera a resposta final usando o contexto extraído do grafo."""
        if not tuplas_grafo:
            return (
                "Não encontrei informações relevantes suficientes no grafo para responder a essa pergunta."
            )

        prompt = self.generate_answer(question, tuplas_grafo)
        try:
            response = self.model.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return "Ocorreu um erro ao tentar gerar a resposta com o modelo de linguagem."