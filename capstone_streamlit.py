# capstone_streamlit.py
# Run with: streamlit run capstone_streamlit.py

import os
import re
import ast
import math
import uuid
import json
import operator
from typing import TypedDict, List, Optional, Annotated

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Engineering Study Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --text: #e2e8f0;
    --muted: #64748b;
    --success: #10b981;
    --warning: #f59e0b;
    --border: rgba(0,212,255,0.15);
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Title area */
.agent-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.agent-sub {
    color: var(--muted);
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1.5rem;
}

/* Chat messages */
.msg-user {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-left: 15%;
    font-size: 0.93rem;
}
.msg-assistant {
    background: linear-gradient(135deg, #0f1f35, #0a0e1a);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-right: 10%;
    font-size: 0.93rem;
}
.msg-label {
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.msg-label-user { color: #a78bfa; }
.msg-label-assistant { color: var(--accent); }

/* Metadata badges */
.meta-row { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.badge {
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 8px;
    border-radius: 20px;
    border: 1px solid;
}
.badge-route { border-color: #7c3aed; color: #a78bfa; background: rgba(124,58,237,0.1); }
.badge-source { border-color: #0891b2; color: var(--accent); background: rgba(0,212,255,0.08); }
.badge-faith-high { border-color: #10b981; color: #34d399; background: rgba(16,185,129,0.1); }
.badge-faith-mid { border-color: #f59e0b; color: #fbbf24; background: rgba(245,158,11,0.1); }
.badge-faith-low { border-color: #ef4444; color: #f87171; background: rgba(239,68,68,0.1); }

/* Input box */
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), #4f46e5) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px);
}

/* Sidebar elements */
.sidebar-section {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    font-size: 0.83rem;
}
.sidebar-section h4 {
    color: var(--accent);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 8px 0;
    font-family: 'JetBrains Mono', monospace;
}

/* Scrollable chat area */
.chat-container {
    max-height: 65vh;
    overflow-y: auto;
    padding-right: 4px;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Selectbox */
div[data-baseweb="select"] {
    background: var(--surface2) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {"id": "doc_001", "topic": "Logic Gates",
     "text": "Logic gates are the fundamental building blocks of digital circuits. The basic gates are AND, OR, NOT, NAND, NOR, XOR, and XNOR. An AND gate outputs 1 only when all inputs are 1. An OR gate outputs 1 when at least one input is 1. A NOT gate (inverter) outputs the complement of its input. NAND and NOR gates are called universal gates because any Boolean function can be implemented using only NAND gates or only NOR gates. An XOR gate outputs 1 when inputs differ; XNOR outputs 1 when inputs are equal. Gates are implemented using transistors in CMOS technology. Gate propagation delay is the time for a signal change at the input to appear at the output. Typical CMOS gate delay is 0.1–1 nanosecond."},
    {"id": "doc_002", "topic": "Flip-Flops",
     "text": "Flip-flops are bistable memory elements that store one bit of data. The SR flip-flop has Set (S) and Reset (R) inputs; the forbidden state is S=R=1. The D flip-flop captures the value of D on the clock edge, eliminating the forbidden state. The JK flip-flop improves on SR by toggling when J=K=1. The T flip-flop toggles its output when T=1. Flip-flops are edge-triggered, responding only on rising or falling clock edges, unlike latches which are level-sensitive. Setup time is the minimum time data must be stable before the clock edge; hold time is the minimum stable time after. Violation of setup/hold time causes metastability. Flip-flops are used in registers, counters, and state machines."},
    {"id": "doc_003", "topic": "Number Systems",
     "text": "Digital systems use several number systems. Binary (base-2) uses digits 0 and 1. Octal (base-8) uses digits 0–7. Hexadecimal (base-16) uses 0–9 and A–F. To convert binary to decimal, sum powers of 2 for each '1' bit. Two's complement is the standard for signed integers: invert all bits and add 1 to get the negative. For an 8-bit system, range is -128 to +127. BCD (Binary-Coded Decimal) encodes each decimal digit in 4 bits. IEEE 754 standard represents floating-point numbers with sign, exponent, and mantissa fields. Single precision uses 32 bits; double uses 64. Gray code is a binary sequence where adjacent values differ by one bit, used in rotary encoders to avoid switching errors."},
    {"id": "doc_004", "topic": "Boolean Algebra",
     "text": "Boolean algebra is the mathematical foundation of digital logic. Key laws: Identity (A+0=A, A·1=A), Null (A+1=1, A·0=0), Idempotent (A+A=A, A·A=A), Complement (A+A'=1, A·A'=0), Double negation (A''=A). De Morgan's theorems: (A·B)'=A'+B' and (A+B)'=A'·B'. Karnaugh maps (K-maps) provide a graphical method to simplify Boolean expressions. Groups of 1s in powers of 2 (1,2,4,8) are circled to find minimal Sum of Products (SOP) expressions. Quine-McCluskey algorithm is a tabular method for minimization. SOP (Sum of Products) and POS (Product of Sums) are standard forms. Canonical forms include minterms (SOP) and maxterms (POS)."},
    {"id": "doc_005", "topic": "RISC-V Basics",
     "text": "RISC-V is an open-standard ISA (Instruction Set Architecture) based on RISC principles, developed at UC Berkeley. It has a fixed 32-bit instruction width (base ISA). RISC-V has 32 general-purpose registers (x0–x31); x0 is hardwired to zero. Instruction types: R-type (register), I-type (immediate), S-type (store), B-type (branch), U-type (upper immediate), J-type (jump). Key instructions: ADD, SUB, AND, OR, XOR, SLL (shift left logical), SRL (shift right logical), LW (load word), SW (store word), BEQ (branch equal), JAL (jump and link). RISC-V uses load/store architecture: only LW/SW access memory. The program counter (PC) advances by 4 bytes each instruction. RISC-V supports extensions: M (multiply), F (float), D (double), A (atomic)."},
    {"id": "doc_006", "topic": "CPU Architecture",
     "text": "A CPU consists of the ALU, Control Unit, registers, and buses. The classic von Neumann architecture has a shared bus for instructions and data. Harvard architecture uses separate instruction and data memories. The fetch-decode-execute cycle: PC fetches instruction, decoder interprets opcode, ALU executes, result is written back. Pipelining overlaps these stages to improve throughput; a 5-stage pipeline has IF, ID, EX, MEM, WB stages. Hazards: structural (resource conflict), data (RAW, WAW, WAR dependencies), and control (branch misprediction). Forwarding/bypassing resolves data hazards. Branch prediction reduces control hazard penalties. Superscalar CPUs issue multiple instructions per cycle. Out-of-order execution improves utilization. Modern CPUs use branch predictors with >95% accuracy."},
    {"id": "doc_007", "topic": "Memory Hierarchy",
     "text": "Memory hierarchy balances speed, cost, and capacity. From fastest to slowest: registers (< 1 ns), L1 cache (1–4 ns, 32–64 KB), L2 cache (4–12 ns, 256 KB–1 MB), L3 cache (10–40 ns, 4–32 MB), RAM (60–100 ns, GBs), SSD (50–150 µs), HDD (5–10 ms). Cache uses temporal and spatial locality principles. Cache hit: data found in cache. Cache miss: must fetch from lower level. Mapping policies: direct-mapped, set-associative, fully associative. Replacement policies: LRU (Least Recently Used), FIFO, random. Write policies: write-through (write to cache and memory simultaneously) and write-back (write to memory only on eviction). Virtual memory uses pages (4 KB typical) and a TLB to speed address translation."},
    {"id": "doc_008", "topic": "Finite State Machines (FSM)",
     "text": "A Finite State Machine (FSM) models sequential logic with a finite number of states. Components: states, inputs, outputs, transition function, output function, initial state. Mealy machine: outputs depend on both current state and input. Moore machine: outputs depend only on the current state. State diagrams show states as circles and transitions as arrows labeled with input/output. State tables enumerate next-state and output for all state-input combinations. State encoding assigns binary codes to states; one-hot encoding uses one bit per state. FSM design steps: define states → state diagram → state table → minimization → state encoding → flip-flop selection → derive equations → implement circuit. FSMs are used in traffic controllers, vending machines, protocol controllers, and parsers."},
    {"id": "doc_009", "topic": "Timing Diagrams",
     "text": "Timing diagrams are waveform representations showing signal values over time. The x-axis is time; the y-axis shows logic HIGH (1) or LOW (0). Clock signals are periodic with defined period T, frequency f=1/T, and duty cycle. Propagation delay (tpd) is the time from input change to output change. Setup time (tsu): data must be stable this long before clock edge. Hold time (th): data must remain stable this long after clock edge. Clock-to-Q delay (tCQ): time from clock edge to stable output in a flip-flop. Maximum clock frequency: f_max = 1 / (tCQ + tpd + tsu). Glitches are unwanted transient pulses caused by unequal gate delays. Hazards (static-1, static-0, dynamic) in combinational circuits cause glitches and can be eliminated by adding redundant gates based on K-map analysis."},
    {"id": "doc_010", "topic": "Combinational Circuits",
     "text": "Combinational circuits produce outputs solely based on current inputs, with no memory. Key combinational circuits: Half adder: adds two 1-bit inputs, produces sum (XOR) and carry (AND). Full adder: adds three inputs (A, B, Cin), produces sum and Cout. Ripple carry adder chains N full adders for N-bit addition. Multiplexer (MUX): selects one of 2^n inputs using n select lines. A 4:1 MUX has 2 select lines. Demultiplexer (DEMUX): routes one input to one of 2^n outputs. Encoder: converts 2^n inputs to n-bit binary output. Decoder: converts n-bit binary to one of 2^n outputs. Comparator: determines if A=B, A>B, or A<B. Barrel shifter: shifts input by variable amount in single gate delay. Combinational circuits are analyzed using truth tables and Boolean expressions."},
]

# ─────────────────────────────────────────────
# RAG + GRAPH (cached so it's only built once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Building knowledge base…")
def load_agent():
    embedding_fn = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    texts = [d["text"] for d in KNOWLEDGE_BASE]
    metas = [{"id": d["id"], "topic": d["topic"]} for d in KNOWLEDGE_BASE]
    vs = Chroma.from_texts(
        texts=texts, embedding=embedding_fn, metadatas=metas,
        collection_name="eng_kb_ui", persist_directory="./chroma_db_ui",
    )

    # ── State ──
    class AgentState(TypedDict):
        question: str
        messages: Annotated[List[BaseMessage], operator.add]
        route: str
        retrieved: List[str]
        sources: List[str]
        tool_result: Optional[str]
        answer: str
        faithfulness: float
        eval_retries: int

    # ── Helpers ──
    CALC_KW = ["calculate","compute","how much","evaluate","sum","multiply","divide","power","sqrt","add","subtract"]
    OOS = ["recipe","cook","movie","cricket","football","weather","news","stock","politics","celebrity"]
    SAFE = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    SAFE["abs"] = abs

    def safe_eval(expr):
        clean = re.sub(r"[^0-9\+\-\*/\.\(\)\^% a-zA-Z_]", "", expr).replace("^","**")
        try:
            return str(round(eval(clean, {"__builtins__": {}}, SAFE), 6))  # noqa: S307
        except Exception as e:
            return f"Error: {e}"

    def extract_answer(question, chunks, tool_result):
        if tool_result and tool_result != "None":
            return f"Calculation result: {tool_result}"
        if not chunks:
            return "I don't have enough information in my knowledge base to answer this."
        qw = set(re.findall(r"\w+", question.lower())) - {
            "what","is","are","the","a","an","how","explain","describe","define",
            "does","do","in","of","for","and","or","with","to","it","this","that"}
        scored = []
        for chunk in chunks:
            for sent in re.split(r"(?<=[.!?])\s+", chunk):
                ov = len(qw & set(re.findall(r"\w+", sent.lower())))
                if ov: scored.append((ov, sent))
        if not scored:
            return chunks[0][:350]
        scored.sort(key=lambda x: x[0], reverse=True)
        return " ".join(s for _, s in scored[:3])

    # ── Nodes ──
    def memory_node(s):
        return {"messages": [HumanMessage(content=s["question"])]}

    def router_node(s):
        q = s["question"].lower()
        if any(kw in q for kw in OOS): return {"route": "skip"}
        has_num = bool(re.search(r"\d", q))
        has_op  = bool(re.search(r"[\+\-\*/\^]", q))
        has_kw  = any(kw in q for kw in CALC_KW)
        if has_num and (has_kw or has_op): return {"route": "tool"}
        return {"route": "retrieve"}

    def retrieval_node(s):
        res = vs.similarity_search_with_score(s["question"], k=3)
        return {"retrieved": [d.page_content for d, _ in res],
                "sources":   [d.metadata.get("topic","?") for d, _ in res]}

    def skip_node(s):
        return {"answer": "I'm specialised in Digital Design, Computer Architecture, and RISC-V topics. That question is outside my domain — please ask about logic gates, flip-flops, CPU architecture, RISC-V, FSMs, etc.",
                "retrieved": [], "sources": [], "faithfulness": 1.0}

    def tool_node(s):
        m = re.search(r"[\d\s\+\-\*/\.\(\)\^%]+", s["question"])
        expr = m.group(0).strip() if m else s["question"]
        return {"tool_result": safe_eval(expr)}

    def answer_node(s):
        ans = extract_answer(s["question"], s.get("retrieved",[]), s.get("tool_result") or "None")
        return {"answer": ans}

    def eval_node(s):
        ans = s.get("answer",""); ctx = " ".join(s.get("retrieved",[]))
        retries = s.get("eval_retries", 0)
        if not ctx or not ans: return {"faithfulness": 0.5, "eval_retries": retries}
        stop = {"i","the","a","an","is","are","was","were","be","to","of","and","or","in","on"}
        aw = set(re.findall(r"\w+", ans.lower())) - stop
        cw = set(re.findall(r"\w+", ctx.lower()))
        score = len(aw & cw) / len(aw) if aw else 0.5
        return {"faithfulness": round(score, 3), "eval_retries": retries + 1}

    def save_node(s):
        return {"messages": [AIMessage(content=s["answer"])]}

    def route_dec(s): return s["route"]
    def eval_dec(s):
        return "retry" if s["faithfulness"] < 0.3 and s.get("eval_retries",0) < 2 else "done"

    # ── Graph ──
    mem = MemorySaver()
    g = StateGraph(AgentState)
    for name, fn in [("memory",memory_node),("router",router_node),
                     ("retrieval",retrieval_node),("skip",skip_node),
                     ("tool",tool_node),("answer",answer_node),
                     ("eval",eval_node),("save",save_node)]:
        g.add_node(name, fn)

    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_conditional_edges("router", route_dec,
                            {"retrieve":"retrieval","tool":"tool","skip":"skip"})
    g.add_edge("retrieval", "answer")
    g.add_edge("tool", "answer")
    g.add_edge("skip", "save")
    g.add_edge("answer", "eval")
    g.add_conditional_edges("eval", eval_dec, {"retry":"retrieval","done":"save"})
    g.add_edge("save", END)
    return g.compile(checkpointer=mem)


agent = load_agent()

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {role, content, meta}
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Study Assistant")
    st.markdown("---")

    st.markdown(f"""
    <div class="sidebar-section">
        <h4>Session Info</h4>
        <b>Thread ID:</b> <code>{st.session_state.thread_id}</code><br>
        <b>Questions asked:</b> {st.session_state.question_count}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>📚 Knowledge Domains</h4>
        • Logic Gates<br>• Flip-Flops<br>• Number Systems<br>• Boolean Algebra<br>
        • RISC-V Basics<br>• CPU Architecture<br>• Memory Hierarchy<br>
        • Finite State Machines<br>• Timing Diagrams<br>• Combinational Circuits
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>🔧 Routing Logic</h4>
        <span style="color:#a78bfa">● retrieve</span> — domain questions<br>
        <span style="color:#00d4ff">● tool</span> — math / calculator<br>
        <span style="color:#64748b">● skip</span> — out-of-scope
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**💡 Try asking:**")
    sample_questions = [
        "What are NAND and NOR universal gates?",
        "Explain D flip-flop setup time",
        "What is RISC-V R-type format?",
        "Calculate 2^8 + 144",
        "Describe 5-stage CPU pipeline",
        "What is LRU replacement policy?",
    ]
    for sq in sample_questions:
        if st.button(sq, key=f"sq_{sq[:20]}", use_container_width=True):
            st.session_state["prefill"] = sq

    st.markdown("---")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.chat_history = []
        st.session_state.question_count = 0
        st.rerun()

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem 0">
    <div class="agent-title">⚡ Engineering Study Assistant</div>
    <div class="agent-sub">B.Tech ECE/CS · Digital Design · CPU Architecture · RISC-V</div>
</div>
""", unsafe_allow_html=True)

# Chat history display
chat_html = ""
for turn in st.session_state.chat_history:
    if turn["role"] == "user":
        chat_html += f"""
        <div class="msg-user">
            <div class="msg-label msg-label-user">You</div>
            {turn['content']}
        </div>"""
    else:
        meta = turn.get("meta", {})
        route = meta.get("route", "")
        sources = meta.get("sources", [])
        faith = meta.get("faithfulness", None)

        # badges
        route_badge = f'<span class="badge badge-route">route: {route}</span>' if route else ""
        src_badges = "".join(f'<span class="badge badge-source">{s}</span>' for s in sources[:2])
        if faith is not None:
            fc = "badge-faith-high" if faith >= 0.5 else ("badge-faith-mid" if faith >= 0.3 else "badge-faith-low")
            faith_badge = f'<span class="badge {fc}">faith: {faith:.2f}</span>'
        else:
            faith_badge = ""

        badges = f'<div class="meta-row">{route_badge}{src_badges}{faith_badge}</div>' if (route_badge or src_badges or faith_badge) else ""
        chat_html += f"""
        <div class="msg-assistant">
            <div class="msg-label msg-label-assistant">Assistant</div>
            {turn['content']}
            {badges}
        </div>"""

if chat_html:
    st.markdown(f'<div class="chat-container">{chat_html}</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #475569;">
        <div style="font-size:3rem; margin-bottom:1rem">⚡</div>
        <div style="font-size:1.1rem; font-weight:700; color:#94a3b8">Ask anything about Digital Design or Computer Architecture</div>
        <div style="font-size:0.85rem; margin-top:0.5rem; font-family: 'JetBrains Mono', monospace">
            Logic Gates · Flip-Flops · RISC-V · CPU Pipeline · FSM · Boolean Algebra
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Input row
prefill = st.session_state.pop("prefill", "")
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "Ask a question…",
        value=prefill,
        key="user_input",
        label_visibility="collapsed",
        placeholder="e.g. What is a full adder? or calculate sqrt(256) + 10",
    )
with col2:
    send = st.button("Send ➤", use_container_width=True)

# ─────────────────────────────────────────────
# AGENT INVOCATION
# ─────────────────────────────────────────────
if send and user_input.strip():
    q = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "content": q})
    st.session_state.question_count += 1

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = {
        "question": q, "messages": [], "route": "",
        "retrieved": [], "sources": [], "tool_result": None,
        "answer": "", "faithfulness": 0.0, "eval_retries": 0,
    }

    with st.spinner("Thinking…"):
        result = agent.invoke(initial_state, config=config)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result.get("answer", "No answer generated."),
        "meta": {
            "route": result.get("route", ""),
            "sources": result.get("sources", []),
            "faithfulness": result.get("faithfulness"),
        },
    })
    st.rerun()
