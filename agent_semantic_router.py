from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

EMBED_MODEL="Qwen/Qwen3-Embedding-0.6B"

encoder = HuggingFaceEncoder(name=EMBED_MODEL)

print(f"Semantic Router with Qwen3 Embedding Model {EMBED_MODEL} is ready to use!")


stock_analysis_route = Route(
    name="stock_analysis",
    utterances=[
        "What is Apple's P/E ratio?",
        "Show me Tesla's market cap",
        "What are NVIDIA's earnings per share?",
        "How much revenue did Microsoft make last year?",
        "What is Amazon's 52-week high?",
        "Give me the fundamentals for Google stock",
        "What is the current valuation of Meta?",
        "How profitable is JPMorgan?",
        "What's the EPS for Netflix?",
        "Tell me about Berkshire Hathaway's financials",
    ],
)

market_news_route = Route(
    name="market_news",
    utterances=[
        "What is the latest news about Apple?",
        "What happened with Tesla recently?",
        "Any recent announcements from the Fed?",
        "What did the SEC say about crypto this week?",
        "Tell me about recent layoffs in tech",
        "What's happening with the banking sector?",
        "Any earnings surprises this quarter?",
        "What are analysts saying about NVIDIA?",
        "Recent news on interest rate decisions",
        "What happened with the SVB collapse?",
    ],
)

general_finance_route = Route(
    name="general_finance",
    utterances=[
        "What is compound interest?",
        "Explain how the stock market works",
        "What is a P/E ratio?",
        "What is dollar cost averaging?",
        "How do ETFs work?",
        "What is the difference between a stock and a bond?",
        "Explain what inflation means",
        "What is a hedge fund?",
        "How does the Federal Reserve control interest rates?",
        "What is diversification in investing?",
    ],
)

routes = [stock_analysis_route, market_news_route, general_finance_route]

sr = SemanticRouter(
    encoder=encoder,
    routes=routes,
    auto_sync="local", # Store the route vector locally in Memory
)

print("\nSemanticRouter ready!")
print(f"Routes: {[r.name for r in routes]}")
print(f"{50*'='}\n")

test_queries = [
    "What is Apple's current P/E ratio?",
    "Tell me about the latest Fed rate decision",
    "How does compound interest work?",
    "Give me NVIDIA's revenue figures",
    "What happened to Silicon Valley Bank?",
    "What is dollar cost averaging?",
    "Can you explain what a hedge fund is?",
]
print("Routing test queries before optimization:\n")
for query in test_queries:
    result = sr(query)
    route_name = result.name if result.name else "No route matched"
    print(f"  Query: '{query}'")
    print(f"  → Route: {route_name}\n")
print(f"{50*'='}\n")


# When you instantiate the SemanticRouter with routes and utterances:

# It calculates vector embeddings for all your utterances (e.g., "What is Apple's P/E ratio?") using the Qwen model.
# When you input a test query (e.g., "What is Apple's current P/E ratio?"), 
# the router embeds your query and calculates the cosine similarity between your query's vector and the utterances' vectors.
# By default, SemanticRouter uses a fixed, hardcoded similarity threshold 
# (e.g., 0.30 or 0.82 depending on the encoder) to decide if a query matches a route.

# The Problem: A single default threshold is a guessing game. It is often:

# Too low: Out-of-scope queries (like "What is the weather in Mumbai?") might accidentally cross the threshold and get routed to a finance category.
# Overlapping: A query like "Give me NVIDIA's revenue figures" has overlapping words with news topics and general finance. Without optimized boundaries, 
# it got misrouted to market_news because that route's utterances were numerically "closer" in the vector space than the default stock_analysis threshold.


train_data = [
    # stock_analysis
    ("What is the P/E ratio for Amazon?",          "stock_analysis"),
    ("Show me TSLA's market cap",                   "stock_analysis"),
    ("How much did Google earn last quarter?",      "stock_analysis"),
    ("What is Microsoft's EPS?",                    "stock_analysis"),
    ("What is the book value of JP Morgan stock?",  "stock_analysis"),
    ("AAPL 52 week high",                           "stock_analysis"),
    ("Tell me NVIDIA's annual revenue",             "stock_analysis"),
    ("Is Berkshire Hathaway overvalued?",           "stock_analysis"),
    ("What's the dividend yield for Coca-Cola?",    "stock_analysis"),
    ("Netflix earnings per share this year",        "stock_analysis"),
    # market_news
    ("What are the latest headlines for Tesla?",    "market_news"),
    ("Did Apple release any news today?",           "market_news"),
    ("What did Jerome Powell say this week?",       "market_news"),
    ("Recent announcements from Amazon",            "market_news"),
    ("What is going on with bank stocks lately?",   "market_news"),
    ("Any SEC filings from Meta recently?",         "market_news"),
    ("Breaking news about the crypto market",       "market_news"),
    ("What happened at the last FOMC meeting?",     "market_news"),
    ("NVIDIA earnings announcement reaction",       "market_news"),
    ("Tech layoffs news 2024",                      "market_news"),
    # general_finance
    ("Can you explain what a mutual fund is?",      "general_finance"),
    ("What does ROI stand for?",                    "general_finance"),
    ("Explain the concept of market liquidity",     "general_finance"),
    ("What is the difference between a Roth and traditional IRA?", "general_finance"),
    ("How do central banks work?",                  "general_finance"),
    ("What is a bear market?",                      "general_finance"),
    ("Explain options trading basics",              "general_finance"),
    ("What is the efficient market hypothesis?",    "general_finance"),
    ("How does inflation affect stock prices?",     "general_finance"),
    ("What is a balance sheet?",                    "general_finance"),
    # None — out of scope (OOS)
    ("What is the weather in Mumbai tomorrow?",     None),
    ("Write me a poem about the ocean",             None),
    ("Who won the cricket World Cup?",              None),
    ("Recommend a good Italian restaurant",         None),
    ("How do I learn to code in Python?",           None),
]
X_train, y_train = zip(*train_data)


print("Default route thresholds:")
thresholds_before = sr.get_thresholds()
for route_name, threshold in thresholds_before.items():
    print(f"  {route_name}: {threshold:.4f}")
accuracy_before = sr.evaluate(X=list(X_train), y=list(y_train))
print(f"\nAccuracy BEFORE optimization: {accuracy_before * 100:.2f}%")
print(f"{50*'='}\n")

# fit() method trains and optimizes data, 
# The router converts your query into vector(using the embedding model)
# Then it calculates the similarity between your query's vector and the pre-defined utterances in your routes
# if the highest similarity score is above threshold, it matches that route. otherwise it returns None

print("Running router.fit() to optimize thresholds...")
sr.fit(X=list(X_train), y=list(y_train))
print("\nOptimized thresholds:")
thresholds_after = sr.get_thresholds()
for name, val in thresholds_after.items():
    print(f"  {name}: {val:.4f}")
accuracy_after = sr.evaluate(X=list(X_train), y=list(y_train))
print(f"Accuracy AFTER optimization: {accuracy_after * 100:.2f}%")
print(f"Improvement: +{(accuracy_after - accuracy_before) * 100:.2f} percentage points")
print(f"{50*'='}\n")

# Testing on completely unseen queries
print("Testing on completely unseen queries:\n")
test_data = [
    # stock_analysis
    ("How is Apple stock performing fundamentally?",     "stock_analysis"),
    ("What is Tesla's trailing twelve months revenue?",  "stock_analysis"),
    ("MSFT price to earnings",                           "stock_analysis"),
    # market_news
    ("Any breaking news about interest rates?",          "market_news"),
    ("What did Elon Musk say about Tesla recently?",     "market_news"),
    ("What are traders saying about gold prices?",       "market_news"),
    # general_finance
    ("Explain what short selling means",                 "general_finance"),
    ("What is a stop-loss order?",                       "general_finance"),
    ("How is GDP calculated?",                           "general_finance"),
    # Out of scope
    ("What movies are showing this weekend?",            None),
    ("Give me a recipe for banana bread",                None),
]
X_test, y_test = zip(*test_data)
test_accuracy = sr.evaluate(X=list(X_test), y=list(y_test))
print(f"Test set accuracy (unseen queries): {test_accuracy * 100:.2f}%\n")
print("Per-query test results:")
for query, expected in test_data:
    result = sr(query)
    got = result.name if result.name else None
    status = "✅" if got == expected else "❌"
    print(f"  {status} Query: '{query}'")
    print(f"     Expected: {expected} | Got: {got}")