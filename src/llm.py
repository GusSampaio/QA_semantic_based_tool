from dotenv import load_dotenv
import os
from langchain_mistralai import ChatMistralAI
from src.settings import AppSettings, missing_required_env

load_dotenv() 

class LLM:
    def __init__(self, settings: AppSettings):
        self.model = self.build_llm(settings)
    
    def build_llm(self,settings: AppSettings) -> ChatMistralAI:
        if not missing_required_env(settings):
            return ChatMistralAI(
                model=settings.chat_model,
                api_key=settings.mistral_api_key,
                temperature=0.2,
        )
    
    def build_context(self, tuplas_grafo: list) -> str:
        context = ""
        for origem, papel, destino in tuplas_grafo:
            context += f"{origem} --{papel}--> {destino}\n"
        return context.strip()

    def generate_answer(self, question: str, context: str) -> str:
        prompt =  f"""Você é um assistente de tarefas de pergunta e resposta. Use as partes seguintes do contexto recuperado por um grafo (representada por tuplas) para responder a pergunta. O contexto considera papéis semânticos anotados
Pergunta: {question}
Contexto: {context}"""
        print("Prompt para LLM:\n", prompt)
        return prompt
    

    def answer_question_with_llm(
        self,
        question: str,
        tuplas_grafo: str,
    ) -> str:
        """Generate the final answer using only retrieved MongoDB context."""
        if not tuplas_grafo:
            return (
                "Não encontrei informações relevantes suficientes no grafo para responder a essa pergunta."
            )

        # context = self.build_context(tuplas_grafo)

        prompt = self.generate_answer(question, tuplas_grafo)
        try:
            response = self.model.invoke(prompt)
            print("***RESPOSTA: ",response.content)
            return response.content.strip()
        except Exception as e:
            print("Erro ao gerar resposta com LLM:", e)
            return "Ocorreu um erro ao tentar gerar a resposta com o modelo de linguagem."