# Projeto PLN - Grafo Semântico

Aplicação em Python com Streamlit para extrair relações semânticas de textos em português e representá-las como um grafo.

O projeto utiliza técnicas de Processamento de Linguagem Natural para identificar eventos, participantes e relações em frases, permitindo visualizar essas informações em uma estrutura de grafo.

## Objetivo

O objetivo do projeto é transformar textos em português em uma representação visual baseada em grafo semântico.

A aplicação permite:

- inserir um texto;
- processar frases com spaCy;
- extrair frames semânticos;
- construir um grafo com nós e arestas;
- visualizar o grafo gerado;
- consultar o grafo com apoio de um modelo de linguagem.

## Tecnologias utilizadas

- Python
- Streamlit
- spaCy
- pt_core_news_sm
- NetworkX
- Matplotlib
- Pandas
- LangChain
- Mistral AI
- uv

## Estrutura do projeto

```text
.
├── app.py
├── src/
│   ├── auxiliares.py
│   ├── extracoes.py
│   ├── frames.py
│   ├── grafo.py
│   ├── llm.py
│   └── settings.py
├── doc/
│   └── regras_utilizadas.txt
├── pyproject.toml
├── uv.lock
├── .python-version
└── README.md
```

## Principais arquivos

- `app.py`: ponto de entrada da aplicação Streamlit.
- `src/auxiliares.py`: funções auxiliares para limpeza, segmentação e normalização de texto.
- `src/extracoes.py`: organiza o processo de extração dos frames.
- `src/frames.py`: contém as regras de extração semântica.
- `src/grafo.py`: constrói, consulta e desenha o grafo.
- `src/llm.py`: integra o projeto com um modelo de linguagem.
- `src/settings.py`: carrega configurações e variáveis de ambiente.
- `doc/regras_utilizadas.txt`: documentação das regras linguísticas usadas no projeto.

## Como executar

### 1. Instalar o uv

Caso ainda não tenha o uv instalado, consulte a documentação oficial:

https://docs.astral.sh/uv/

### 2. Instalar as dependências

Na raiz do projeto, execute:

```bash
uv sync
```

### 3. Executar a aplicação

```bash
uv run python -m streamlit run app.py
```

Depois, acesse o endereço exibido no terminal.

## Variáveis de ambiente

Para usar a funcionalidade com modelo de linguagem, crie um arquivo `.env` na raiz do projeto:

```env
MISTRAL_API_KEY=sua_chave_aqui
CHAT_MODEL=mistral-small-latest
```

A variável `MISTRAL_API_KEY` é necessária para gerar respostas usando a API da Mistral.

## Funcionamento geral

O fluxo principal da aplicação é:

```text
Texto do usuário
→ limpeza e separação em frases
→ processamento com spaCy
→ extração de frames semânticos
→ construção do grafo
→ visualização no Streamlit
→ pergunta ao grafo com apoio do LLM
```

## Exemplo de uso

1. Abra a aplicação.
2. Insira ou edite o texto no campo principal.
3. Clique em **Processar texto**.
4. Visualize o grafo gerado.
5. Faça uma pergunta sobre o conteúdo processado.

## Observações

- O projeto foi desenvolvido para fins acadêmicos.
- O foco principal é a extração e visualização de relações semânticas.
- A qualidade da extração depende da análise sintática feita pelo modelo spaCy.
- A resposta via LLM depende de uma chave válida da Mistral.
