"""Search tool implementation"""
from langchain.tools import Tool


def mock_search(query: str) -> str:
    """
    Mock search function - replace with real search API

    For production, consider using:
    - Tavily API: from langchain_community.tools.tavily_search import TavilySearchResults
    - SerpAPI: from langchain_community.utilities import SerpAPIWrapper
    - DuckDuckGo: from langchain_community.tools import DuckDuckGoSearchRun
    """
    return f"""Search results for '{query}':

    1. Key Finding: Important information about {query}
    2. Statistical Data: Recent statistics related to {query}
    3. Expert Opinion: Perspectives from industry experts on {query}
    4. Recent Development: Latest news and updates about {query}
    5. Case Study: Real-world example involving {query}
    """


def get_search_tool() -> Tool:
    """Create and return the search tool"""
    return Tool(
        name="web_search",
        func=mock_search,
        description="Useful for searching information on the web. Input should be a search query string."
    )

