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
    """Retrieve relevant local knowledge base context for a user query.

    Parameters
    ----------
    query : str
        User question or search phrase.

    Returns
    -------
    str
        A formatted context string built from retrieved documents.
    """
    docs = retriever.invoke(query)
    return build_context(docs)


@tool
def web_search(query, max_results=3):
    """Search the web and return concatenated text snippets.

    Parameters
    ----------
    query : str
        Search query text.
    max_results : int, default=3
        Maximum number of web results to include.

    Returns
    -------
    str
        A newline-separated string of result snippets.
    """
    results = tavily_client.search(query, max_results=max_results)
    snippets = [r["content"] for r in results.get("results", [])]

    return "\n".join(snippets)


def load_chat_model():
    """Create and configure the chat model client used by the agent.

    Returns
    -------
    ChatOpenAI
        A configured ``ChatOpenAI`` instance.
    """
    chat_model = ChatOpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
        model=MODEL,
        max_tokens=512,
        temperature=0
    )
    return chat_model


def build_agent(tools=[rag_tool, web_search], prompt=agent_prompt):
    """Build an agent with the configured chat model and tools.

    Parameters
    ----------
    tools : list, default=[rag_tool, web_search]
        Tool callables available to the agent.
    prompt : str, default=agent_prompt
        System prompt string for agent behavior.

    Returns
    -------
    object
        A LangChain agent ready to be invoked.
    """
    chat_model = load_chat_model()

    agent = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=prompt
    )
    return agent


def invoke_agent(agent, query):
    """Invoke the agent with a single user message.

    Parameters
    ----------
    agent : object
        The initialized LangChain agent.
    query : str
        User input text.

    Returns
    -------
    dict
        The raw agent response payload.
    """
    question = HumanMessage(
        content=query
    )
    return agent.invoke({"messages": [question]})


def format_tools(response):
    """Extract and format tool outputs from an agent response.

    Parameters
    ----------
    response : dict
        Agent response dictionary containing ``messages``.

    Returns
    -------
    str
        A human-readable string with collected tool outputs.
    """
    tool_msgs = [i.content for i in response['messages'] if type(i) == ToolMessage]
    if len(tool_msgs) == 0:
        msgs = "None"
    else:
        msgs = "\n".join(tool_msgs)
    return "Tool Output:\n" + msgs


def format_response(response):
    """Return the final assistant message content from a response.

    Parameters
    ----------
    response : dict
        Agent response dictionary containing ``messages``.

    Returns
    -------
    str
        The text content of the last message.
    """
    return response['messages'][-1].content
