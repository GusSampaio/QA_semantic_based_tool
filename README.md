# Projeto PLN - Grafo Semântico

Aplicação em Python com Streamlit para extrair relações semânticas de textos em português e representá-las como um grafo.

O projeto utiliza técnicas de Processamento de Linguagem Natural para identificar eventos, participantes e relações em frases, permitindo visualizar essas informações em uma estrutura de grafo.

O desenvolvimento (em andamento) do modelo BERT utilizado parcialmente neste projeto, pode ser encontrado em [neste link](https://github.com/GusSampaio/brazilian_semantic_bert).

## Objetivo

### Funcionamento Gramatical e Lógico da Extração

O objetivo deste projeto é transformar textos livres em português em uma representação visual estruturada na forma de grafos semânticos, automatizando a descoberta de conhecimento linguístico. A aplicação centraliza todo esse fluxo em uma interface interativa que permite ao usuário inserir um texto, processar suas frases por meio do framework spaCy e extrair frames semânticos detalhados. A partir desses dados, o sistema constrói uma rede de nós e arestas que é renderizada visualmente na tela, oferecendo ainda uma camada de inteligência que permite ao usuário consultar as informações do grafo gerado conversacionalmente com o apoio de um modelo de linguagem (LLM).

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
O uso de uma API para o hugging face é opcional, visto que a chamada para o modelo funciona sem ela.

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

## Metodologia e Métricas de Avaliação

A validação do projeto foi realizada utilizando um conjunto de teste extraído do córpus [Porttinari-base Propbank](https://sites.google.com/icmc.usp.br/poetisa/porttinari-base-propbank) (versão clássica, disponibilizado [neste link do Hugging Face](https://huggingface.co/datasets/GusSampaio/pbp-srl-test-predictions/tree/main)). A avaliação comparou o desempenho global e por papel semântico entre o modelo baseado em Deep Learning (BERT) e a abordagem heurística por regras desenvolvida neste repositório.

Enquanto o modelo BERT apresentou alta robustez generalista — alcançando um F1-Score global de 76.21 e excelentes métricas nos argumentos centrais (F1 superior a 92% para Arg0 e Arg1), a abordagem por regras funcionou como uma linha de base (baseline) leve e determinística. 

F1-score medido
| Papel | Modelo BERT  | Estratégia por Regras |
| :--- | :---: | :---: |
| **Global** | 76.21 | 0.4197 |
| **ARG0** (Agente) | 92.66 | 51.31 |
| **ARG1** (Paciente) | 93.08 | 39.12 |
| **ARGM-LOC** (Lugar) | 75.44 | 43.48 |
| **ARGM-TMP** (Tempo) | 87.00 | 15.14  |
