import os
import sqlite3
import yfinance as yf
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
import wikipedia as wiki
import requests


load_dotenv()


llm = ChatGroq(
    model=os.getenv("GROQ_MODEL"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


@tool
def get_stock_fundamentals(ticker: str) -> dict:
    """Get current stock price, P/E ratio, market cap, and revenue growth for a ticker."""

    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "price": info.get("currentPrice"),
        "pe_ratio": info.get("trailingPE"),
        "market_cap": info.get("marketCap"),
        "revenue_growth": info.get("revenueGrowth"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
    }

search = SerpAPIWrapper(
    serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
    params={
        "tbm": "nws",    # Search Google News only
        "tbs": "qdr:d",  # Within the past day only
    }
)

@tool
def search_news(query: str) -> str:
    """ Search last-24h Google news via SerpAPI.  Return news results with URLs."""
    return search.run(f"{query} news")


_original_wiki_wrapper = WikipediaAPIWrapper()
@tool
def robust_wikipedia_search(query: str) -> str:
    """Search Wikipedia robustly, handling disambiguation and page errors."""
    try:
        return _original_wiki_wrapper.run(query)
    except requests.exceptions.JSONDecodeError as e:
        return f"Wikipedia search failed for '{query}' due to invalid JSON. Error: {e}"
    except wiki.exceptions.PageError:
        return f"Wikipedia page not found for '{query}'."
    except wiki.exceptions.DisambiguationError as e:
        return f"Wikipedia query '{query}' is ambiguous. Options: {e.options}"
    except Exception as e:
        return f"Unexpected error during Wikipedia search: {e}"
tools = [
    get_stock_fundamentals,
    YahooFinanceNewsTool(),
    robust_wikipedia_search,
    search_news
]


SYSTEM_PROMPT = """
You are FinBot, an expert equity research analyst with deep knowledge of financial markets,
valuation methodologies, and macroeconomic trends.
## Task
Given a stock ticker or company name, produce a concise, structured analyst brief that helps users evaluate the investment. Do not give buy/sell advice. Present data-driven signals only.
"""

conn = sqlite3.connect("finance_agent.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

my_finance_agent_persistent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer  # <-- SqliteSaver enables persistent memory
)


#Define a Thread id so the memory is saved for a specific thread
config = {"configurable": {"thread_id": "apple-nvda-thread"}}

response1 = my_finance_agent_persistent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Give me a Quick analyst brief on APPLE. Is now a goot time to buy?"}
        ]
    },
    config
)

print(response1) # prints the loops of tool calls and the final response
print(50*"=")
print(response1['messages'][-1].content) # prints the final response

response2 = my_finance_agent_persistent.invoke(
    {
        "messages": [
            {"role": "user", "content": "How does it compare to NVDA on valuation?"}
        ]
    },
    config
)

# print(response2) # prints the loops of tool calls and the final response
print(50*"=")
print(response2['messages'][-1].content) # prints the final response