# This is a custom agent which will call the tools directly without using LangChain's Agent
# This is to understand what is happening inside the LangChain's Agent
# It will use Groq's SDK directly, LangChain just provides a wrapper around the SDK
# The Agent class handles many things automatically, which we need to handle manually in this custom agent

import os
import yfinance as yf
from dotenv import load_dotenv
import json
import groq
import wikipedia as wiki
import requests

load_dotenv()

groq = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
groq_model = os.getenv("GROQ_MODEL")

def get_stock_fundamentals(ticker: str) -> dict:
    """Get current stock price, P/E ratio, market cap, and revenue growth for a ticker."""
    
    try:
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
    except Exception as e:
        return {"error": f"Failed to get stock fundamentals for {ticker}: {e}"}

def robust_wikipedia_search(query: str) -> str:
    """Robust Wikipedia search - handles errors and returns informative messages."""

    try:
        return wiki.summary(query, sentences=3)
    except requests.exceptions.JSONDecodeError as e:
        return f"Wikipedia search failed for '{query}' due to invalid JSON response. The Wikipedia API might have returned an error or malformed data. Error: {e}"
    except wiki.exceptions.PageError:
        return f"Wikipedia page not found for '{query}'."
    except wiki.exceptions.DisambiguationError as e:
        return f"Wikipedia query '{query}' is ambiguous. Please be more specific. Options: {e.options}"
    except Exception as e:
        return f"An unexpected error occurred during Wikipedia search for '{query}': {e}"

def search_news(query: str) -> str:
    """ Search last-24h Google news via SerpAPI.  Return news results with URLs."""

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "SerpAPI key is not set. Please set it in the .env file."
    params = {
        "engine": "google",
        "q": f"{query} news",
        "tbm": "nws",
        "tbs": "qdr:d",
        "api_key": api_key,
    }
    # Call the API endpoint for search
    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()
    news_results = data.get("news_results") # returns a list of news results
    if not news_results:
        return "No news found for the query."
    
    # COnvert the list of dictionaries to a clean formatted string, LLM can parse JSON, but pass a string.
    # Because of Token usage - A string is more compact than a JSON
    format_news = []
    for news in news_results[:4]:
        title = news.get("title")
        link = news.get("link")
        source = news.get("source")
        date = news.get("date")
        snippet = news.get("snippet")
        format_news.append(f"Title: {title}\nLink: {link}\nSource: {source}\nDate: {date}\nSnippet: {snippet}\n")
    # Join the list to make a single string
    return "\n".join(format_news)

# Defining tool schemas for Groq's API, create a dictionary with name and description which is OpenAI Function/Tool Calling format
# All the major LLM API's hav standardiezed this format. And it is widely adopted.
# You pass this tools list to the LLM and model uses the descriptions to decide if it needs to call a tool and 
# then formats the arguments as a JSON string matching your paramters 

tools = []

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_fundamentals",
            "description": "Get current stock price, P/E ratio, market cap, and revenue growth for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g. AAPL, TSLA, MSFT)"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "robust_wikipedia_search",
            "description": "Search Wikipedia to get summaries of concepts, companies, or people.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search Wikipedia for."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search the last 24h Google news headlines and snippets for a given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic or company name to search news for."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ReAct Execution Loop - set a limit on the loop of iterations otherwise it goes in infite tool calling loop

SYSTEM_PROMPT = """
You are FinBot, an expert equity research analyst.
Produce a concise, structured analyst brief. Never fabricate numbers.
Always base your responses on the retrieved data.
Do not give BUY/SELL signals
"""

def run_Agent(user_query):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    for loop in range(5):

        llm_response = groq.chat.completions.create(
            model=groq_model,
            messages=messages,
            tools=tools,
        )
        # print(f"LLM Response: {llm_response}")
        # print("="*50)
        
        assitant_message = llm_response.choices[0].message
        # print(f"Assitant Message: {assitant_message}")
        # print("*"*50)
        messages.append(assitant_message)

        # Check if the assistant wants to call any tools
        if not assitant_message.tool_calls:
            # If no tool calls, it means the assistant has a final answer!
            return assitant_message.content
            
        # If it does want to call tools, loop over them and run them
        for tool_call in assitant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            # Execute the correct function based on tool_name
            if tool_name == "get_stock_fundamentals":
                # Call get_stock_fundamentals with the ticker argument
                result = get_stock_fundamentals(tool_args.get("ticker"))
                
            elif tool_name == "robust_wikipedia_search":
                # Call robust_wikipedia_search with the query argument
                result = robust_wikipedia_search(tool_args.get("query"))
                
            elif tool_name == "search_news":
                # Call search_news with the query argument
                result = search_news(tool_args.get("query"))
                
            else:
                # Safety fallback
                result = f"Error: Tool '{tool_name}' not found."
            
            # Append the tool result back to the messages list
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": str(result)
            })
            
            # Print a quick log so you can see it working in the console:
            print(f"Executed Tool: {tool_name} with args {tool_args}")
    

    

if __name__ == "__main__":
    print(run_Agent("Give me a Quick analyst brief on APPLE"))

    