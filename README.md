# Ingestão e Busca Semântica com LangChain e Postgres (pgVector) · PT/EN

> **PT:** Starter minimalista que ingere um PDF em um banco PostgreSQL com pgVector e realiza buscas semânticas via LangChain, respondendo no terminal com base **apenas** no conteúdo do PDF.
>
> **EN:** Minimal starter that ingests a PDF into PostgreSQL with pgVector and performs semantic search via LangChain, answering in the terminal using **only** the PDF content.

---

## ✅ Objetivo (PT)
- **Ingestão:** Ler um PDF, dividir em chunks e salvar embeddings no Postgres (pgVector).
- **Busca:** Perguntar via CLI e receber respostas **somente** com base no conteúdo do PDF.

## ✅ Goal (EN)
- **Ingestion:** Read a PDF, split into chunks, and store embeddings in Postgres (pgVector).
- **Search:** Ask via CLI and get answers **only** based on the PDF content.

---

## 🧰 Tecnologias / Stack
- **Python**
- **LangChain**
- **PostgreSQL + pgVector**
- **Docker & Docker Compose** (para subir o banco / to start the DB)

### Pacotes recomendados (PT/EN)
- Split: `from langchain_text_splitters import RecursiveCharacterTextSplitter`
- Embeddings (OpenAI): `from langchain_openai import OpenAIEmbeddings`
- Embeddings (Gemini): `from langchain_google_genai import GoogleGenerativeAIEmbeddings`
- PDF: `from langchain_community.document_loaders import PyPDFLoader`
- Vetor store: `from langchain_postgres import PGVector`

---

## 📁 Estrutura do projeto / Project structure

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── prompts
│   ├── qa_prompt_en.txt  #pronpt in English
│   └── qa_prompt_pt.txt  #prompt in Portuguese
├── src/
│   ├── ingest.py      # Ingestão do PDF / PDF ingestion
│   ├── common.py      # Utilidades / Utility functions
│   ├── search.py      # Busca no banco vetorial / Semantic search
│   ├── chat.py        # CLI para interação / CLI chat
├── document.pdf       # Adicione seu PDF aqui / Place your PDF here
└── README.md
```

---

## 🌱 Setup

### 1) VirtualEnv (Unix-like)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1) VirtualEnv (Windows PowerShell)
```powershell
py -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Banco de Dados / Database
Suba o Postgres com pgVector:
```bash
docker compose up -d
```

### 3) Configuração / Configuration
Copie `.env.example` para `.env` e preencha as chaves:
```bash
cp .env.example .env  # Windows: copy .env.example .env
```

- Se usar **OpenAI**:
  - `EMBEDDINGS_PROVIDER=openai`
  - `LLM_PROVIDER=openai`
  - `OPENAI_API_KEY=...`
  - (padrões: `text-embedding-3-small`, `gpt-5-nano`)

- Se usar **Gemini**:
  - `EMBEDDINGS_PROVIDER=gemini`
  - `LLM_PROVIDER=gemini`
  - `GOOGLE_API_KEY=...`
  - (padrões: `models/embedding-001`, `gemini-2.5-flash-lite`)

> Observação: os nomes de modelos acima seguem sua sugestão. Se sua conta não suportar tais modelos, ajuste no `.env`.

### 4) Adicione o PDF
Coloque seu arquivo como `./document.pdf`.

---

## ▶️ Execução / Run

### Ingestão do PDF
```bash
python src/ingest.py --pdf ./document.pdf
```

### Chat no terminal
```bash
python src/chat.py
```
#### Obs: Caso queira iniciar o chat sem fazer a ingestão de dados, o chat irá perguntar sem que quer fazer a ingestao de dados.

Exemplo no CLI (PT):
```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.
```

Perguntas fora do contexto (PT):
```
PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

---

## 🧪 Como funciona (resumo) / How it works (summary)
- O `ingest.py` lê o PDF, divide em **chunks de 1000** com **overlap 150**, gera embeddings com o provedor escolhido e persiste em uma coleção no Postgres/pgVector.
- O `search.py` executa `similarity_search_with_score(query, k=10)` na coleção.
- O `chat.py` monta o prompt com o **CONTEXTO** (top K) e chama a LLM. As regras impedem alucinação: se não houver evidência no contexto, responde a negativa padronizada.

---

## 🪪 Licença / License
MIT
