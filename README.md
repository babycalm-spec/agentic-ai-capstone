# Engineering Study Assistant Agent
### Agentic AI Capstone Project

---

## Project Structure

```
├── engineering_study_agent.py   ← Main agent (Colab / Python)
├── capstone_streamlit.py        ← Streamlit web UI
└── README.md
```

---

## Quick Start

### Step 1 — Install dependencies

```bash
pip install langchain langchain-community langgraph chromadb \
    sentence-transformers streamlit
```

### Step 2 — Run in Google Colab

Open `engineering_study_agent.py` and run all sections top-to-bottom.  
The script will:
- Build the ChromaDB vector store
- Test retrieval
- Compile the LangGraph agent
- Run 9 test questions automatically

### Step 3 — Run the Streamlit UI

```bash
streamlit run capstone_streamlit.py
```

For Google Colab, add this cell to expose via ngrok:

```python
!pip install pyngrok
from pyngrok import ngrok
import subprocess, threading

def run_app():
    subprocess.run(["streamlit", "run", "capstone_streamlit.py",
                    "--server.port", "8501", "--server.headless", "true"])

threading.Thread(target=run_app, daemon=True).start()
import time; time.sleep(5)
public_url = ngrok.connect(8501)
print("Streamlit URL:", public_url)
```

---

## Architecture

```
User Question
     │
     ▼
[memory_node] ──► Adds question to message history
     │
     ▼
[router_node] ──► Decides: retrieve / tool / skip
     │
     ├── retrieve ──► [retrieval_node] ──► ChromaDB similarity search
     │                       │
     ├── tool ────► [tool_node] ────────► Safe calculator (eval)
     │                       │
     └── skip ────► [skip_node] ─────────────────────────┐
                             │                            │
                             ▼                            │
                       [answer_node] ◄───────────────────┘
                             │
                             ▼
                        [eval_node] ──► faithfulness score
                             │
                     ┌───────┴────────┐
                   retry           done
                     │               │
                     ▼               ▼
               [retrieval]       [save_node] ──► END
```

---

## Nodes

| Node | Purpose |
|------|---------|
| `memory_node` | Appends question to LangGraph message history |
| `router_node` | Routes to retrieve / tool / skip |
| `retrieval_node` | Fetches top-3 chunks from ChromaDB |
| `skip_node` | Handles out-of-scope questions gracefully |
| `tool_node` | Safe calculator using `eval` with math whitelist |
| `answer_node` | Generates answer from context or tool result only |
| `eval_node` | Computes faithfulness score (word overlap) |
| `save_node` | Saves turn; adds AIMessage to history |

---

## Knowledge Base Topics

1. Logic Gates
2. Flip-Flops
3. Number Systems
4. Boolean Algebra
5. RISC-V Basics
6. CPU Architecture
7. Memory Hierarchy
8. Finite State Machines (FSM)
9. Timing Diagrams
10. Combinational Circuits

---

## Test Questions Covered

| # | Question | Type |
|---|----------|------|
| 1 | What are the basic logic gates? | Normal |
| 2 | Difference between Mealy and Moore machines | Normal |
| 3 | RISC-V R-type instruction format | Normal |
| 4 | What is 2 + 2 * 10? | Tool (calculator) |
| 5 | Calculate sqrt(144) + 5^2 | Tool (calculator) |
| 6 | You mentioned RISC-V. What registers does it use? | Memory (follow-up) |
| 7 | What is the best cricket team? | Out-of-scope |
| 8 | Memory hierarchy fastest to slowest | Normal |
| 9 | Maximum clock frequency formula | Normal |

---

## State Schema

```python
class AgentState(TypedDict):
    question:     str           # Current user question
    messages:     List[BaseMessage]  # Full conversation history
    route:        str           # "retrieve" | "tool" | "skip"
    retrieved:    List[str]     # Context chunks from ChromaDB
    sources:      List[str]     # Source topics
    tool_result:  Optional[str] # Calculator output
    answer:       str           # Final answer
    faithfulness: float         # 0.0 – 1.0 score
    eval_retries: int           # Retry counter (max 2)
```

---

## Notes

- **No LLM API key required** — uses rule-based answer extraction from retrieved context
- **To upgrade**: replace `_extract_answer()` in `answer_node` with a call to
  `anthropic` / `openai` / `google-generativeai` for production-grade answers
- ChromaDB is persisted to `./chroma_db` so embeddings are reused across runs
- Thread IDs enable per-user memory isolation via LangGraph's `MemorySaver`
