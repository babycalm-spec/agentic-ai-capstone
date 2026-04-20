# ============================================================
# ENGINEERING STUDY ASSISTANT AGENT - COMPLETE CAPSTONE
# ============================================================

# ============================================================
# SECTION 1: INSTALL COMMANDS
# ============================================================
# Run these in a Colab cell:
"""
!pip install -q langchain langchain-community langgraph chromadb \
    sentence-transformers streamlit pyngrok
"""

# ============================================================
# SECTION 2: IMPORTS
# ============================================================
import os
import re
import ast
import math
import uuid
import json
import operator
from typing import TypedDict, List, Optional, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import chromadb

# ============================================================
# SECTION 3: KNOWLEDGE BASE (10 documents)
# ============================================================
KNOWLEDGE_BASE = [
    {
        "id": "doc_001",
        "topic": "Logic Gates",
        "text": (
            "Logic gates are the fundamental building blocks of digital circuits. "
            "The basic gates are AND, OR, NOT, NAND, NOR, XOR, and XNOR. "
            "An AND gate outputs 1 only when all inputs are 1. "
            "An OR gate outputs 1 when at least one input is 1. "
            "A NOT gate (inverter) outputs the complement of its input. "
            "NAND and NOR gates are called universal gates because any Boolean "
            "function can be implemented using only NAND gates or only NOR gates. "
            "An XOR gate outputs 1 when inputs differ; XNOR outputs 1 when inputs "
            "are equal. Gates are implemented using transistors in CMOS technology. "
            "Gate propagation delay is the time for a signal change at the input to "
            "appear at the output. Typical CMOS gate delay is 0.1–1 nanosecond."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Flip-Flops",
        "text": (
            "Flip-flops are bistable memory elements that store one bit of data. "
            "The SR flip-flop has Set (S) and Reset (R) inputs; the forbidden state "
            "is S=R=1. The D flip-flop captures the value of D on the clock edge, "
            "eliminating the forbidden state. The JK flip-flop improves on SR by "
            "toggling when J=K=1. The T flip-flop toggles its output when T=1. "
            "Flip-flops are edge-triggered, responding only on rising or falling "
            "clock edges, unlike latches which are level-sensitive. Setup time is the "
            "minimum time data must be stable before the clock edge; hold time is the "
            "minimum stable time after. Violation of setup/hold time causes metastability. "
            "Flip-flops are used in registers, counters, and state machines."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Number Systems",
        "text": (
            "Digital systems use several number systems. Binary (base-2) uses digits 0 and 1. "
            "Octal (base-8) uses digits 0–7. Hexadecimal (base-16) uses 0–9 and A–F. "
            "To convert binary to decimal, sum powers of 2 for each '1' bit. "
            "Two's complement is the standard for signed integers: invert all bits and add 1 "
            "to get the negative. For an 8-bit system, range is -128 to +127. "
            "BCD (Binary-Coded Decimal) encodes each decimal digit in 4 bits. "
            "IEEE 754 standard represents floating-point numbers with sign, exponent, "
            "and mantissa fields. Single precision uses 32 bits; double uses 64. "
            "Gray code is a binary sequence where adjacent values differ by one bit, "
            "used in rotary encoders to avoid switching errors."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Boolean Algebra",
        "text": (
            "Boolean algebra is the mathematical foundation of digital logic. "
            "Key laws: Identity (A+0=A, A·1=A), Null (A+1=1, A·0=0), "
            "Idempotent (A+A=A, A·A=A), Complement (A+A'=1, A·A'=0), "
            "Double negation (A''=A). De Morgan's theorems: (A·B)'=A'+B' and (A+B)'=A'·B'. "
            "Karnaugh maps (K-maps) provide a graphical method to simplify Boolean "
            "expressions. Groups of 1s in powers of 2 (1,2,4,8) are circled to find "
            "minimal Sum of Products (SOP) expressions. "
            "Quine-McCluskey algorithm is a tabular method for minimization. "
            "SOP (Sum of Products) and POS (Product of Sums) are standard forms. "
            "Canonical forms include minterms (SOP) and maxterms (POS)."
        ),
    },
    {
        "id": "doc_005",
        "topic": "RISC-V Basics",
        "text": (
            "RISC-V is an open-standard ISA (Instruction Set Architecture) based on RISC "
            "principles, developed at UC Berkeley. It has a fixed 32-bit instruction width "
            "(base ISA). RISC-V has 32 general-purpose registers (x0–x31); x0 is hardwired "
            "to zero. Instruction types: R-type (register), I-type (immediate), S-type "
            "(store), B-type (branch), U-type (upper immediate), J-type (jump). "
            "Key instructions: ADD, SUB, AND, OR, XOR, SLL (shift left logical), "
            "SRL (shift right logical), LW (load word), SW (store word), BEQ (branch equal), "
            "JAL (jump and link). RISC-V uses load/store architecture: only LW/SW access "
            "memory. The program counter (PC) advances by 4 bytes each instruction. "
            "RISC-V supports extensions: M (multiply), F (float), D (double), A (atomic)."
        ),
    },
    {
        "id": "doc_006",
        "topic": "CPU Architecture",
        "text": (
            "A CPU consists of the ALU, Control Unit, registers, and buses. "
            "The classic von Neumann architecture has a shared bus for instructions and data. "
            "Harvard architecture uses separate instruction and data memories. "
            "The fetch-decode-execute cycle: PC fetches instruction, decoder interprets opcode, "
            "ALU executes, result is written back. Pipelining overlaps these stages to improve "
            "throughput; a 5-stage pipeline has IF, ID, EX, MEM, WB stages. "
            "Hazards: structural (resource conflict), data (RAW, WAW, WAR dependencies), "
            "and control (branch misprediction). Forwarding/bypassing resolves data hazards. "
            "Branch prediction reduces control hazard penalties. Superscalar CPUs issue "
            "multiple instructions per cycle. Out-of-order execution improves utilization. "
            "Modern CPUs use branch predictors with >95% accuracy."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Memory Hierarchy",
        "text": (
            "Memory hierarchy balances speed, cost, and capacity. From fastest to slowest: "
            "registers (< 1 ns), L1 cache (1–4 ns, 32–64 KB), L2 cache (4–12 ns, 256 KB–1 MB), "
            "L3 cache (10–40 ns, 4–32 MB), RAM (60–100 ns, GBs), SSD (50–150 µs), "
            "HDD (5–10 ms). Cache uses temporal and spatial locality principles. "
            "Cache hit: data found in cache. Cache miss: must fetch from lower level. "
            "Mapping policies: direct-mapped, set-associative, fully associative. "
            "Replacement policies: LRU (Least Recently Used), FIFO, random. "
            "Write policies: write-through (write to cache and memory simultaneously) "
            "and write-back (write to memory only on eviction). "
            "Virtual memory uses pages (4 KB typical) and a TLB to speed address translation."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Finite State Machines (FSM)",
        "text": (
            "A Finite State Machine (FSM) models sequential logic with a finite number of states. "
            "Components: states, inputs, outputs, transition function, output function, initial state. "
            "Mealy machine: outputs depend on both current state and input. "
            "Moore machine: outputs depend only on the current state. "
            "State diagrams show states as circles and transitions as arrows labeled with input/output. "
            "State tables enumerate next-state and output for all state-input combinations. "
            "State encoding assigns binary codes to states; one-hot encoding uses one bit per state. "
            "FSM design steps: define states → state diagram → state table → minimization "
            "→ state encoding → flip-flop selection → derive equations → implement circuit. "
            "FSMs are used in traffic controllers, vending machines, protocol controllers, and parsers."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Timing Diagrams",
        "text": (
            "Timing diagrams are waveform representations showing signal values over time. "
            "The x-axis is time; the y-axis shows logic HIGH (1) or LOW (0). "
            "Clock signals are periodic with defined period T, frequency f=1/T, and duty cycle. "
            "Propagation delay (tpd) is the time from input change to output change. "
            "Setup time (tsu): data must be stable this long before clock edge. "
            "Hold time (th): data must remain stable this long after clock edge. "
            "Clock-to-Q delay (tCQ): time from clock edge to stable output in a flip-flop. "
            "Maximum clock frequency: f_max = 1 / (tCQ + tpd + tsu). "
            "Glitches are unwanted transient pulses caused by unequal gate delays. "
            "Hazards (static-1, static-0, dynamic) in combinational circuits cause glitches "
            "and can be eliminated by adding redundant gates based on K-map analysis."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Combinational Circuits",
        "text": (
            "Combinational circuits produce outputs solely based on current inputs, with no memory. "
            "Key combinational circuits: "
            "Half adder: adds two 1-bit inputs, produces sum (XOR) and carry (AND). "
            "Full adder: adds three inputs (A, B, Cin), produces sum and Cout. "
            "Ripple carry adder chains N full adders for N-bit addition. "
            "Multiplexer (MUX): selects one of 2^n inputs using n select lines. A 4:1 MUX has "
            "2 select lines. "
            "Demultiplexer (DEMUX): routes one input to one of 2^n outputs. "
            "Encoder: converts 2^n inputs to n-bit binary output. "
            "Decoder: converts n-bit binary to one of 2^n outputs. "
            "Comparator: determines if A=B, A>B, or A<B. "
            "Barrel shifter: shifts input by variable amount in single gate delay. "
            "Combinational circuits are analyzed using truth tables and Boolean expressions."
        ),
    },
]

# ============================================================
# SECTION 4: RAG SETUP
# ============================================================
def build_rag():
    """Build ChromaDB vector store with sentence-transformer embeddings."""
    print("Loading embedding model...")
    embedding_fn = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # Build LangChain Documents
    documents = []
    metadatas = []
    for doc in KNOWLEDGE_BASE:
        documents.append(doc["text"])
        metadatas.append({"id": doc["id"], "topic": doc["topic"]})

    # Create ChromaDB collection via LangChain
    print("Building ChromaDB collection...")
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embedding_fn,
        metadatas=metadatas,
        collection_name="engineering_kb",
        persist_directory="./chroma_db",
    )
    print(f"Collection built with {len(documents)} documents.")
    return vectorstore


def test_retrieval(vectorstore, query: str, k: int = 3):
    """Test retrieval for a given query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    print(f"\nQuery: {query}")
    for doc, score in results:
        print(f"  [{doc.metadata['topic']}] score={score:.4f} | {doc.page_content[:80]}...")
    return results


# Build the vector store (run once)
vectorstore = build_rag()

# Quick retrieval test
test_retrieval(vectorstore, "What are D flip-flops and setup time?")
test_retrieval(vectorstore, "Explain RISC-V instruction types")


# ============================================================
# SECTION 5: STATE DEFINITION
# ============================================================
class AgentState(TypedDict):
    question: str
    messages: Annotated[List[BaseMessage], operator.add]
    route: str                      # "retrieve" | "tool" | "skip"
    retrieved: List[str]            # retrieved context chunks
    sources: List[str]              # source topics
    tool_result: Optional[str]      # calculator result
    answer: str                     # final answer
    faithfulness: float             # 0.0 – 1.0
    eval_retries: int               # retry counter for eval loop


# ============================================================
# SECTION 6: NODE IMPLEMENTATIONS
# ============================================================

# --- 6.1 Memory Node ---
def memory_node(state: AgentState) -> AgentState:
    """Append the current question to the message history."""
    new_msg = HumanMessage(content=state["question"])
    return {"messages": [new_msg]}


# --- 6.2 Router Node ---
CALCULATOR_KEYWORDS = [
    "calculate", "compute", "how much", "what is", "evaluate",
    "sum", "multiply", "divide", "power", "sqrt", "add", "subtract",
    "+", "-", "*", "/", "^", "%",
]
OUT_OF_SCOPE_TOPICS = [
    "recipe", "cook", "movie", "cricket", "football", "weather",
    "news", "stock", "politics", "celebrity", "music", "song",
]

def router_node(state: AgentState) -> AgentState:
    """Route to: retrieve / tool / skip based on question analysis."""
    q = state["question"].lower()

    # Check out-of-scope
    if any(kw in q for kw in OUT_OF_SCOPE_TOPICS):
        return {"route": "skip"}

    # Check if it needs the calculator (has math expression or explicit compute request)
    has_number = bool(re.search(r"\d", q))
    has_calc_kw = any(kw in q for kw in CALCULATOR_KEYWORDS)
    has_math_op = bool(re.search(r"[\+\-\*/\^]", q))

    if has_number and (has_calc_kw or has_math_op):
        return {"route": "tool"}

    return {"route": "retrieve"}


# --- 6.3 Retrieval Node ---
def retrieval_node(state: AgentState) -> AgentState:
    """Retrieve top-3 relevant chunks from ChromaDB."""
    results = vectorstore.similarity_search_with_score(state["question"], k=3)
    chunks = []
    topics = []
    for doc, score in results:
        chunks.append(doc.page_content)
        topics.append(doc.metadata.get("topic", "Unknown"))
    return {"retrieved": chunks, "sources": topics}


# --- 6.4 Skip Node ---
def skip_node(state: AgentState) -> AgentState:
    """Handle out-of-scope questions gracefully."""
    answer = (
        "I'm an Engineering Study Assistant specialised in Digital Design, "
        "Computer Architecture, and RISC-V. Your question appears to be outside "
        "my domain. Please ask about topics like logic gates, flip-flops, CPU "
        "architecture, RISC-V, memory hierarchy, FSMs, or Boolean algebra."
    )
    return {
        "answer": answer,
        "retrieved": [],
        "sources": [],
        "faithfulness": 1.0,
    }


# --- 6.5 Tool Node (Safe Calculator) ---
SAFE_MATH_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
SAFE_MATH_NAMES["abs"] = abs

def safe_eval(expr: str) -> str:
    """Safely evaluate a mathematical expression."""
    # Extract expression: keep digits, operators, parens, dots, spaces
    clean = re.sub(r"[^0-9\+\-\*/\.\(\)\^% a-zA-Z_]", "", expr)
    clean = clean.replace("^", "**")  # support caret for power
    try:
        result = eval(clean, {"__builtins__": {}}, SAFE_MATH_NAMES)  # noqa: S307
        return str(round(result, 6))
    except Exception as e:
        return f"Error: {e}"


def tool_node(state: AgentState) -> AgentState:
    """Extract and evaluate a mathematical expression from the question."""
    q = state["question"]
    # Try to extract the math part
    match = re.search(r"[\d\s\+\-\*/\.\(\)\^%]+", q)
    expr = match.group(0).strip() if match else q
    result = safe_eval(expr)
    return {"tool_result": f"Calculator result: {result}"}


# --- 6.6 Answer Node ---
ANSWER_TEMPLATE = """You are an Engineering Study Assistant for B.Tech students (ECE/CS).
Answer ONLY based on the provided context or tool result.
If the answer is not in the context, say "I don't have enough information in my knowledge base to answer this."
Be concise, accurate, and educational.

Context:
{context}

Tool Result: {tool_result}

Question: {question}

Answer:"""

def answer_node(state: AgentState) -> AgentState:
    """Generate answer strictly from retrieved context or tool result."""
    context_parts = state.get("retrieved", [])
    tool_result = state.get("tool_result") or "None"
    context = "\n\n".join(context_parts) if context_parts else "No context retrieved."

    prompt = ANSWER_TEMPLATE.format(
        context=context,
        tool_result=tool_result,
        question=state["question"],
    )

    # ---- Simple rule-based answer generation (no LLM API needed) ----
    # For production: replace with actual LLM call. For Colab demo, we
    # extract the most relevant sentence from context.
    answer = _extract_answer(state["question"], context_parts, tool_result)

    return {"answer": answer}


def _extract_answer(question: str, chunks: List[str], tool_result: str) -> str:
    """Rule-based answer extractor from context chunks."""
    if tool_result and tool_result != "None":
        return f"Based on the calculation: {tool_result}"

    if not chunks:
        return "I don't have enough information in my knowledge base to answer this."

    q_lower = question.lower()
    # Score sentences by keyword overlap with question
    question_words = set(re.findall(r"\w+", q_lower)) - {"what", "is", "are", "the", "a", "an",
                                                           "how", "explain", "describe", "define",
                                                           "does", "do", "in", "of", "for"}
    best_sentences = []
    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk)
        for sent in sentences:
            sent_words = set(re.findall(r"\w+", sent.lower()))
            overlap = len(question_words & sent_words)
            if overlap > 0:
                best_sentences.append((overlap, sent))

    if not best_sentences:
        return chunks[0][:300] + "..."

    best_sentences.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in best_sentences[:3]]
    return " ".join(top)


# --- 6.7 Eval Node ---
def eval_node(state: AgentState) -> AgentState:
    """Compute faithfulness score: fraction of answer words found in context."""
    answer = state.get("answer", "")
    context = " ".join(state.get("retrieved", []))
    retries = state.get("eval_retries", 0)

    if not context or not answer:
        return {"faithfulness": 0.5, "eval_retries": retries}

    answer_words = set(re.findall(r"\w+", answer.lower()))
    context_words = set(re.findall(r"\w+", context.lower()))
    stop_words = {"i", "the", "a", "an", "is", "are", "was", "were", "be",
                  "to", "of", "and", "or", "in", "on", "at", "for", "with"}
    answer_words -= stop_words

    if not answer_words:
        return {"faithfulness": 0.5, "eval_retries": retries}

    overlap = answer_words & context_words
    score = len(overlap) / len(answer_words)
    return {"faithfulness": round(score, 3), "eval_retries": retries + 1}


# --- 6.8 Save Node ---
CONVERSATION_LOG = []

def save_node(state: AgentState) -> AgentState:
    """Save conversation turn to in-memory log."""
    entry = {
        "question": state["question"],
        "answer": state["answer"],
        "sources": state.get("sources", []),
        "faithfulness": state.get("faithfulness", 0.0),
        "route": state.get("route", "unknown"),
    }
    CONVERSATION_LOG.append(entry)
    # Also add AI response to message history
    ai_msg = AIMessage(content=state["answer"])
    return {"messages": [ai_msg]}


# ============================================================
# SECTION 7: LANGGRAPH GRAPH CONSTRUCTION
# ============================================================

def route_decision(state: AgentState) -> str:
    """Conditional edge: after router, decide next node."""
    return state["route"]   # "retrieve" | "tool" | "skip"


def eval_decision(state: AgentState) -> str:
    """Conditional edge: after eval, retry answer if score too low."""
    if state["faithfulness"] < 0.3 and state.get("eval_retries", 0) < 2:
        return "retry"
    return "done"


def build_graph():
    memory = MemorySaver()
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("skip", skip_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    # Entry point
    graph.set_entry_point("memory")

    # Fixed edges
    graph.add_edge("memory", "router")

    # Conditional routing after router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieval",
            "tool": "tool",
            "skip": "skip",
        },
    )

    # After retrieval → answer
    graph.add_edge("retrieval", "answer")

    # After tool → answer
    graph.add_edge("tool", "answer")

    # Skip → save (bypass eval)
    graph.add_edge("skip", "save")

    # After answer → eval
    graph.add_edge("answer", "eval")

    # Conditional: eval passes or retries
    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {
            "retry": "retrieval",   # re-retrieve and re-answer
            "done": "save",
        },
    )

    # Save → END
    graph.add_edge("save", END)

    return graph.compile(checkpointer=memory)


app = build_graph()
print("Graph compiled successfully!")


# ============================================================
# SECTION 8: TESTING
# ============================================================

def ask(question: str, thread_id: str = "test-thread") -> dict:
    """Run the agent with a question and return the result."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "question": question,
        "messages": [],
        "route": "",
        "retrieved": [],
        "sources": [],
        "tool_result": None,
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0,
    }
    result = app.invoke(initial_state, config=config)
    return result


def run_tests():
    print("\n" + "="*60)
    print("RUNNING TEST SUITE")
    print("="*60)

    test_cases = [
        # (question, thread_id, description)
        ("What are the basic logic gates and their functions?",
         "thread-1", "Normal: Logic Gates"),

        ("Explain the difference between Mealy and Moore machines.",
         "thread-1", "Normal: FSM"),

        ("What is the RISC-V instruction format for R-type instructions?",
         "thread-2", "Normal: RISC-V"),

        ("What is 2 + 2 * 10?",
         "thread-3", "Tool: Basic Calculator"),

        ("Calculate sqrt(144) + 5^2",
         "thread-3", "Tool: Advanced Calculator"),

        ("You mentioned RISC-V earlier. What registers does it use?",
         "thread-2", "Memory: Follow-up on RISC-V"),

        ("What is the best cricket team in the world?",
         "thread-4", "Out-of-scope: Cricket"),

        ("Describe the memory hierarchy from fastest to slowest.",
         "thread-5", "Normal: Memory Hierarchy"),

        ("What is the maximum clock frequency formula in timing diagrams?",
         "thread-5", "Normal: Timing Diagrams"),
    ]

    results_summary = []
    for question, tid, desc in test_cases:
        print(f"\n[{desc}]")
        print(f"Q: {question}")
        res = ask(question, tid)
        print(f"Route  : {res.get('route', 'N/A')}")
        print(f"Sources: {res.get('sources', [])}")
        print(f"Score  : {res.get('faithfulness', 'N/A')}")
        print(f"A: {res.get('answer', '')[:200]}")
        results_summary.append({
            "desc": desc,
            "route": res.get("route"),
            "faithfulness": res.get("faithfulness"),
        })

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results_summary:
        print(f"  {r['desc']:<40} route={r['route']:<10} faith={r['faithfulness']}")

    high_faith = [r for r in results_summary if (r["faithfulness"] or 0) >= 0.3]
    print(f"\nQuestions with faithfulness >= 0.3: {len(high_faith)}/{len(results_summary)}")


run_tests()

print("\n✅ All sections complete. Run capstone_streamlit.py for the web UI.")
