import os
import yfinance as yf
import wikipedia as wiki
from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")


llm = ChatGroq(
    model=groq_model,
    api_key=groq_api_key,
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

tool_augumented_model = llm.bind_tools([get_stock_fundamentals])

response = tool_augumented_model.invoke("What is the current stock price for Apple?")
print(response)

# Below is the response from the LLM which is a OpenAI Tool calling JSON Format

# {
#     'reasoning_content': 'We need to call function get_stock_fundamentals with ticker "AAPL". Then respond with price.',
#     'tool_calls': [{
#         'id': 'fc_7b138ce5-9a3f-4667-b597-bbbf3e06efdf',
#         'function': {
#             'arguments': '{"ticker":"AAPL"}',
#             'name': 'get_stock_fundamentals'
#         },
#         'type': 'function'
#     }]
# }
# response_metadata = {
#     'token_usage': {
#         'completion_tokens': 51,
#         'prompt_tokens': 145,
#         'total_tokens': 196,
#         'completion_time': 0.054004271,
#         'completion_tokens_details': {
#             'reasoning_tokens': 23
#         },
#         'prompt_time': 0.012328035,
#         'prompt_tokens_details': None,
#         'queue_time': 0.628280468,
#         'total_time': 0.066332306
#     },
#     'model_name': 'openai/gpt-oss-safeguard-20b',
#     'system_fingerprint': 'fp_e3febdc4be',
#     'service_tier': 'on_demand',
#     'finish_reason': 'tool_calls',
#     'logprobs': None,
#     'model_provider': 'groq'
# }
# id = 'lc_run--019edd35-0c94-7070-b7ca-08948acaaa44-0'
# tool_calls = [{
#     'name': 'get_stock_fundamentals',
#     'args': {
#         'ticker': 'AAPL'
#     },
#     'id': 'fc_7b138ce5-9a3f-4667-b597-bbbf3e06efdf',
#     'type': 'tool_call'
# }] invalid_tool_calls = [] usage_metadata = {
#     'input_tokens': 145,
#     'output_tokens': 51,
#     'total_tokens': 196,
#     'output_token_details': {
#         'reasoning': 23
#     }
# }

# invoking the tool using for loop to get the result
print(f"{50*'='}\n\n")
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    tool_result = get_stock_fundamentals.invoke(tool_call)
    print(tool_result.content)


# Do a Google Search News and get the headlines

search = SerpAPIWrapper(
    serpapi_api_key=serpapi_api_key,
    params={
        "tbm": "nws",    # Search Google News only
        "tbs": "qdr:d",  # Within the past day only
    }
)

@tool
def search_news(query: str) -> str:
    """ Search last-24h Google news via SerpAPI.  Return news results with URLs."""
    return search.run(f"{query} news")



from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.tools import tool
import wikipedia as wiki
import requests # To catch specific JSONDecodeError if wikipedia library doesn't wrap it fully

# Original Wikipedia API wrapper
_original_wiki_wrapper = WikipediaAPIWrapper()

@tool
def robust_wikipedia_search(query: str) -> str:
    """
    Search Wikipedia robustly.
    Handles JSONDecodeError and other Wikipedia library errors by returning an informative message.
    """

    # wikipedia python package from langchin is very fragile, it can crash the entire agent loop,
    # which is why trying to catch all the exceptions and return an informative message
    try:
        return _original_wiki_wrapper.run(query)
    except requests.exceptions.JSONDecodeError as e:
        return f"Wikipedia search failed for '{query}' due to invalid JSON response. The Wikipedia API might have returned an error or malformed data. Error: {e}"
    except wiki.exceptions.PageError:
        return f"Wikipedia page not found for '{query}'."
    except wiki.exceptions.DisambiguationError as e: # eg: Apple can be a company and fruit, so wiki might be confused
        return f"Wikipedia query '{query}' is ambiguous. Please be more specific. Options: {e.options}"
    except Exception as e: # Catch any other unexpected errors from wikipedia library
        return f"An unexpected error occurred during Wikipedia search for '{query}': {e}"

# Put together all the tools
tools = [
    get_stock_fundamentals,
    YahooFinanceNewsTool(),
    robust_wikipedia_search, # Use the robust wrapper
    search_news
]

SYSTEM_PROMPT = """
You are FinBot, an expert equity research analyst with deep knowledge of financial markets,
valuation methodologies, and macroeconomic trends.
## Task
Given a stock ticker or company name, produce a concise, structured analyst brief that helps users evaluate the investment. Do not give buy/sell advice. Present data-driven signals only.
## Rules
1. Gather data before analysis. Never rely on memory for numbers.
2. If a tool fails or returns empty data, state it and proceed.
3. Never fabricate prices, ratios, or news.
4. Always follow the output format.
5. Flag notable risks or red flags.
## Output Format
**[TICKER] — Analyst Brief**
- 📊 **Fundamentals:** price, P/E, market cap, revenue growth (one line)
- 📈 **Valuation Signal:** OVERVALUED / FAIRLY VALUED / UNDERVALUED + reason
- 📰 **News Sentiment:** bullish / neutral / bearish + key headline
- ⚠️ **Key Risks:** 1–2 bullets
- 🧭 **Outlook:** 1–2 sentence synthesis, no advice
"""

my_finance_agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt = SYSTEM_PROMPT,
    checkpointer=InMemorySaver(), # This Enables a Short Term Memory
)


#Define a Thread id so the memory is saved for a specific thread
config = {"configurable": {"thread_id": "apple-nvda-thread"}}

response1 = my_finance_agent.invoke(
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

response2 = my_finance_agent.invoke(
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

