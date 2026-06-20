# AI Agent & Semantic Router Playground

## 📖 Overview
This repository is a learning playground that demonstrates how to build, extend, and optimise AI agents for financial research. It walks through a series of incremental milestones, showcasing:
- **Agent construction** with LangChain and Groq.
- **Short‑term memory** using LangGraph check‑pointing.
- **Persistent memory** backed by SQLite.
- **Semantic routing** to dispatch queries to specialised sub‑agents.
- **A custom, LangChain‑free implementation** that directly uses the Groq SDK.

---

## 📚 Table of Contents
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage Examples](#usage-examples)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)

---

## 🏗️ Project Structure & Milestones
The project is organised as a linear progression of five milestones. Each milestone adds a new capability and is represented by a dedicated script.

```mermaid
graph TD
    M1[1️⃣ Basic Agent] --> M2[2️⃣ Short‑Term Memory]
    M2 --> M3[3️⃣ Persistent SQLite Memory]
    M3 --> M4[4️⃣ Semantic Router]
    M4 --> M5[5️⃣ Custom Agent (no LangChain)]
```

| Milestone | Script | Key Feature |
|-----------|--------|-------------|
| **1️⃣ Basic Agent** | `basic_agent.py` | Financial analyst bot (`FinBot`) using LangChain, Groq, yfinance, YahooFinanceNewsTool, Wikipedia wrapper, SerpAPI news. |
| **2️⃣ Short‑Term Memory** | `agent_with_memory.py` | In‑memory conversation context with LangGraph `InMemorySaver`. |
| **3️⃣ Persistent SQLite Memory** | `agent_persistant_memory.py` | Durable context across restarts using LangGraph `SqliteSaver` and `finance_agent.db`. |
| **4️⃣ Semantic Router** | `agent_semantic_router.py` | Zero‑latency query routing via `semantic-router` + local Qwen embedding model. |
| **5️⃣ Custom Agent** | `custom_agent.py` | Pure‑Python tool‑calling agent built on the Groq SDK (no LangChain). |

---

## 🛠️ Setup & Installation
The project relies on **uv** – a fast Python package manager and environment tool.

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your‑username>/ai-agent-semantic-router-playground.git
   cd ai-agent-semantic-router-playground
   ```

2. **Create and activate a virtual environment** (uv does this automatically when you run commands, but you can also create one manually):
   ```bash
   uv venv
   source .venv/bin/activate   # on Windows use `.venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Configure environment variables** – copy the template below into a file named `.env` at the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-safeguard-20b   # or any other Groq model you prefer
   SERPAPI_API_KEY=your_serpapi_api_key_here
   ```
   *Both keys are required for the news‑search tools.*

---

## 🚀 Usage Examples
Each milestone can be executed independently. Below are quick commands to try them out.

### 1️⃣ Basic Agent
```bash
uv run python basic_agent.py
```
The agent will ask for a ticker symbol, fetch fundamentals, recent news, and provide a short analysis.

### 2️⃣ Agent with Short‑Term Memory
```bash
uv run python agent_with_memory.py
```
You can have a multi‑turn conversation. The agent remembers prior messages within the same thread.

### 3️⃣ Persistent SQLite Memory
```bash
uv run python agent_persistant_memory.py
```
Run the script, answer the initial prompts, then stop it (`Ctrl‑C`). Running it again will automatically recall the stored context from `finance_agent.db`.

### 4️⃣ Semantic Router
```bash
uv run python agent_semantic_router.py
```
Enter a query – the router will classify it into one of the specialised sub‑agents (`stock_analysis`, `market_news`, `general_finance`) or return *Out‑of‑Scope* for unrelated topics.

### 5️⃣ Custom Agent (no LangChain)
```bash
uv run python custom_agent.py
```
A lightweight agent that demonstrates raw tool‑calling using the Groq SDK. It parses tool requests, runs local Python functions, and feeds results back to the model.

---

## 🤝 Contribution Guidelines
We welcome contributions! Follow these steps to propose improvements:

1. **Fork the repository** on GitHub.
2. **Create a new branch** for your feature or bug‑fix:
   ```bash
   git checkout -b my‑feature‑branch
   ```
3. **Make your changes** – keep code style consistent (PEP 8) and add or update documentation as needed.
4. **Run tests** (if any) and ensure the scripts still execute with `uv run`.
5. **Commit with a clear message**:
   ```bash
   git commit -m "feat: add X feature"   # or "fix: resolve Y bug"
   ```
6. **Push to your fork** and open a Pull Request against the `main` branch.
7. **Fill the PR template** – describe the problem, the solution, and any relevant screenshots or logs.

### Code of Conduct
Please be respectful and inclusive. Harassment or discriminatory language will not be tolerated.

---

## 📄 License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

*Happy hacking!*
