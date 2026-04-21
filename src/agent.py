import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from tavily import TavilyClient
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import VECTOR_STORE_DIR
from src.vectorstore import load_vectorstore
from src.rag_pipeline import semantic_retriever, build_context
from src.prompts import agent_prompt

load_dotenv()

MODEL = "gpt-4o-mini"
tavily_client = TavilyClient()

vectorstore = load_vectorstore(VECTOR_STORE_DIR)
retriever = semantic_retriever(vectorstore, k=3)


@tool
def rag_tool(query: str) -> str:
    """Retrieve product context for a user query."""
    docs = retriever.invoke(query)
    return build_context(docs)


@tool
def web_search(query, max_results=3):
    """Search the web for information"""
    results = tavily_client.search(query, max_results=max_results)
    snippets = [r["content"] for r in results.get("results", [])]

    return "\n".join(snippets)


def load_chat_model():
    chat_model = ChatOpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
        model=MODEL,
        max_tokens=512,
        temperature=0
    )
    return chat_model


def build_agent(tools=[rag_tool, web_search], prompt=agent_prompt):
    chat_model = load_chat_model()

    agent = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=prompt
    )
    return agent


def invoke_agent(agent, query):
    question = HumanMessage(
        content=query
    )
    return agent.invoke({"messages": [question]})


def format_tools(response):
    tool_msgs = [i.content for i in response['messages'] if type(i) == ToolMessage]
    if len(tool_msgs) == 0:
        msgs = "None"
    else:
        msgs = "\n".join(tool_msgs)
    return "Tool Output:\n" + msgs


def format_response(response):
    return response['messages'][-1].content
