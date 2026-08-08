"""
agent.py — a LangGraph agent with TWO tools (documents + data).
The agent decides on its own which tool(s) to call. For a combined question it
will call BOTH in sequence and synthesise the answer — the full agentic pattern.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from rag_tool import search_documents            # tool 1: documents (RAG)
from sql_tool import query_operational_data       # tool 2: data (NL-to-SQL)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# The agent now has BOTH tools and chooses which to use per question.
tools = [search_documents, query_operational_data]

agent = create_agent(llm, tools)


def ask(question: str):
    result = agent.invoke({"messages": [("user", question)]})
    for msg in result["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    print("\n########## DOCUMENT question ##########")
    ask("At what gearbox oil temperature must a turbine be shut down?")
    print("\n########## DATA question ##########")
    ask("How many assets had a FAULT status?")
    print("\n########## COMBINED question ##########")
    ask("Which assets had a FAULT, and what should I do about a fault?")
