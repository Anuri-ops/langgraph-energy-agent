"""
hello_graph.py — the smallest possible LangGraph.
Shows the three concepts: STATE, NODE, EDGE. No LLM, no tools yet.
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. STATE — the shared "notebook" that flows through the graph.
#    Here it holds two fields: a name (input) and a greeting (output).
class State(TypedDict):
    name: str
    greeting: str


# 2. NODE — a step of work. Takes the current state, returns an update to it.
#    This node reads state["name"] and writes state["greeting"].
def greet_node(state: State) -> State:
    name = state["name"]
    return {"greeting": f"Hello, {name}! This graph is working."}


# 3. BUILD THE GRAPH — add the node, then connect it with EDGES.
builder = StateGraph(State)
builder.add_node("greet", greet_node)     # register the node under the name "greet"
builder.add_edge(START, "greet")          # edge: start -> greet
builder.add_edge("greet", END)            # edge: greet -> end

graph = builder.compile()                 # turn the blueprint into a runnable graph


# RUN IT
if __name__ == "__main__":
    result = graph.invoke({"name": "Anuri"})   # give it the starting state
    print(result)
    print(result["greeting"])