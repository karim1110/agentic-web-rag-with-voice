import os, httpx

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

def web_search(query: str, top_k: int = 5):
    api_key = os.getenv("SEARCH_API_KEY")
    search_provider = os.getenv("SEARCH_PROVIDER", "brave")
    
    if not api_key:
        print(f"[web.search] Warning: SEARCH_API_KEY not set, returning empty results")
        return []

    if search_provider == "brave":
        return brave_search(query, top_k, api_key)

    return []  # fallback


def brave_search(query, top_k, api_key):
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }

    params = {
        "q": query,
        "count": top_k
    }

    try:
        with httpx.Client(timeout=20) as client:
            resp = client.get(BRAVE_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"[web.search] Brave error: {e}")
        print(f"[web.search] Query: '{query}', API Key set: {bool(api_key)}")
        return []

    # Normalize output for the agent
    web_results = data.get("web", {}).get("results", [])
    if not web_results:
        print(f"[web.search] Brave returned no results for query: '{query}'")
    
    results = []
    for item in web_results[:top_k]:
        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "snippet": item.get("description"),
            "profile": item.get("profile", {}).get("name"),
            "price": None,
            "availability": None
        })

    print(f"[web.search] Returned {len(results)} results for query: '{query}'")
    return results
