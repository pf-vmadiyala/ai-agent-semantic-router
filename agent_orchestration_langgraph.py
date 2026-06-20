from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from IPython.display import Image, display
import os

load_dotenv()

# Creating a very simple state
class SimpleState(TypedDict):
    topic: str
    draft: str
    score: int
    approved: bool

# Message-aware state(for conversational agents)
# list[BaseMessage]: Specifies that the variable is a list of message objects.
# add_messages: Tells LangGraph: "Whenever a node returns a list of messages, do not overwrite the old list. Instead, pass the old list and the new list to the add_messages function, which merges them."
# What add_messages does:
# If you return a new message, add_messages appends it to the end of the list.
# If you return a message that has the same ID as a message already in the list, it updates that message in-place (which is how LangGraph handles editing or correcting messages).
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    iteration: int

groq_llm = ChatGroq(
    model=os.getenv("GROQ_MODEL"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2
)

# A node in langraph is a any python function that operate on the data(state) or perform some action.

# Node 1:
def generate_draft(state: SimpleState) -> dict:
    """Writes a first draft on the given topic."""
    response = groq_llm.invoke(f"Write a paragraph about: {state['topic']}")
    return {"draft": response.content, "iteration": 1}

# Node 1 — Simple LLM call
def generate_draft(state: SimpleState) -> dict:
    """Writes a first draft on the given topic."""
    response = groq_llm.invoke(f"Write a short paragraph about: {state['topic']}")
    return {"draft": response.content, "iteration": 1}

# Node 2 — Scoring agent
def score_draft(state: SimpleState) -> dict:
    """Scores the draft quality from 1–10."""
    prompt = f"""
    Rate this paragraph from 1–10 for clarity. Respond with ONLY a number.
    Paragraph: {state['draft']}
    """
    score = int(groq_llm.invoke(prompt).content.strip())
    return {"score": score}

# Node 3 — Revision agent
def revise_draft(state: SimpleState) -> dict:
    """Improves a low-scoring draft."""
    response = groq_llm.invoke(
        f"Improve this paragraph for clarity:\n{state['draft']}"
    )
    return {"draft": response.content, "iteration": state["iteration"] + 1}

builder = StateGraph(SimpleState)
builder.add_node("generate", generate_draft)
builder.add_node("score", score_draft)
builder.add_node("revise", revise_draft)

# Edges connect nodes and tell LangGraph what happens after each step.
# START is a keyword in langgraph that represents the starting point of the graph.
# END is a keyword in langgraph that represents the ending point of the graph.

builder.add_edge(START, "generate")
builder.add_edge("generate", "score")
builder.add_edge("revise", "score") 

# Conditional Edges let us create dynamic control flow that depends on the output of a node.
# Conditional Edges call a routing function that looks at state and returns the name of the next node.END

def quality_gate(state: SimpleState) -> str:
    """Route based on quality score."""
    if state["score"] >= 7:
        return "approved"
    elif state["iteration"] >= 3:
        return "give_up"
    else:
        return "needs_work"

builder.add_conditional_edges(
    "score",
    quality_gate,
    {
        "approved":   END,
        "give_up":    END,
        "needs_work": "revise",
    }
)

# compile and Visualize & Run the Graph

graph = builder.compile()
# display(Image(graph.get_graph().draw_mermaid_png())) 
# or
graph.get_graph().print_ascii() # PRints the graph in the terminal using grandalf

result = graph.invoke({"topic": "quantam computing", "draft": "", "score": 0, "approved": False})
print(result["draft"])
