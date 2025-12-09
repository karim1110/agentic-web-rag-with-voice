import json
from rapidfuzz import fuzz
from graph.llm_client import get_llm_client, load_prompt


def reconcile(rag_items, web_items):
    """Match RAG results with web results using fuzzy matching."""
    out = []
    matched_web_urls = set()
    
    # First pass: match RAG items with web items
    for r in rag_items:
        match = None
        score_best = 0
        for w in (web_items or []):
            s = fuzz.token_set_ratio(r.get("title", ""), w.get("title", ""))
            if s > score_best:
                score_best, match = s, w
        
        # Check for price conflict
        conflict = None
        if match and r.get("price") and match.get("price"):
            try:
                rag_price = float(r["price"])
                web_price = float(match["price"])
                diff_pct = abs(rag_price - web_price) / rag_price * 100
                if diff_pct > 10:
                    conflict = f"price_diff_{diff_pct:.1f}%"
            except (ValueError, TypeError):
                pass
        
        if match and score_best > 80:
            matched_web_urls.add(match.get("url"))
        
        out.append({
            "primary": r,
            "web_match": match if score_best > 80 else None,
            "score": score_best,
            "conflict": conflict,
            "source_type": "rag"
        })
    
    # Second pass: add unmatched web items as standalone entries
    for w in (web_items or []):
        if w.get("url") not in matched_web_urls:
            out.append({
                "primary": w,
                "web_match": None,
                "score": 0,
                "conflict": None,
                "source_type": "web_only"
            })
    
    return out


def answer(state):
    """
    Answerer Agent: Synthesize grounded response using LLM.
    """
    rag = (state.get("evidence") or {}).get("rag", [])
    web = (state.get("evidence") or {}).get("web", [])
    plan = state.get("plan") or {}
    transcript = state.get("transcript", "")
    
    # Check for empty evidence
    if not rag and not web:
        state.update(
            answer="I couldn't find any products matching those criteria. Try broadening your search or adjusting filters.",
            citations=[]
        )
        state.setdefault("log", []).append({"node": "answerer", "status": "no_results"})
        return state
    
    # Load system prompt
    system_prompt = load_prompt("system_answerer.md")
    
    # Prepare evidence summary - show BOTH sources separately so LLM can choose
    evidence_text = "## Evidence Retrieved:\n\n"
    
    # Show RAG results first
    if rag:
        evidence_text += "### Private Catalog (RAG) - Top 5:\n"
        for i, r in enumerate(rag[:5], 1):
            evidence_text += f"{i}. **{r.get('title', 'Unknown')}**\n"
            evidence_text += f"   - Doc ID: {r.get('doc_id') or r.get('sku')}\n"
            evidence_text += f"   - Category: {r.get('category', 'N/A')}\n"
            evidence_text += f"   - Brand: {r.get('brand') or 'N/A'}\n"
            evidence_text += f"   - Price: ${r.get('price', 'N/A')}\n"
            evidence_text += f"   - Rating: {r.get('rating', 'N/A')}\n"
            evidence_text += f"   - Ingredients: {r.get('ingredients', 'N/A')[:100]}\n\n"
    
    # Show web results separately
    if web:
        evidence_text += "### Web Search Results - Top 5:\n"
        for i, w in enumerate(web[:5], 1):
            evidence_text += f"{i}. **{w.get('title', 'Unknown')}**\n"
            evidence_text += f"   - URL: {w.get('url')}\n"
            evidence_text += f"   - Snippet: {w.get('snippet', 'N/A')[:200]}\n"
            evidence_text += f"   - Price: {w.get('price') or 'Not available'}\n\n"
    
    # Add decision guidance
    evidence_text += "\n**IMPORTANT**: Check if the RAG results are RELEVANT to the user query. "
    evidence_text += "If RAG results are off-topic (wrong product category), use ONLY the web results in your answer."
    
    # Prepare messages
    context = f"""
User query: {transcript}

{evidence_text}

Synthesize a concise voice response (≤15 seconds / ~50 words) with proper citations.
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context}
    ]
    
    # Call LLM
    try:
        llm = get_llm_client()
        response = llm.chat(messages, temperature=0.4, max_tokens=300)
        
        # Build citations from all available sources (LLM will use what it needs)
        citations = []
        
        # Add RAG citations
        for r in rag[:5]:
            citations.append({
                "doc_id": r.get("doc_id") or r.get("sku"),
                "source": "private",
                "title": r.get("title", "")[:100]
            })
        
        # Add web citations
        for w in web[:5]:
            citations.append({
                "url": w.get("url"),
                "source": "web",
                "title": w.get("title", "")[:100]
            })
        
        answer_text = response.strip()
        
    except Exception as e:
        # Fallback to template-based answer
        lines = []
        citations = []
        
        # Use web results if available, otherwise RAG
        items_to_use = web[:3] if web else rag[:3]
        
        for i, item in enumerate(items_to_use, 1):
            title = item.get('title', 'Product')[:60]
            if 'url' in item:
                # Web result
                lines.append(f"{i}. {title} (see link)")
                citations.append({"url": item.get("url"), "source": "web"})
            else:
                # RAG result
                price = f"${item.get('price')}" if item.get('price') else "price N/A"
                lines.append(f"{i}. {title} — {price}")
                citations.append({
                    "doc_id": item.get("doc_id") or item.get("sku"),
                    "source": "private"
                })
        
        answer_text = "Here are options that fit your request. " + " ".join(lines) + " See details on your screen."
        
        state.setdefault("log", []).append({
            "node": "answerer",
            "warning": "llm_fallback",
            "error": str(e)
        })
    
    # Update state
    state.update(answer=answer_text, citations=citations)
    state.setdefault("log", []).append({
        "node": "answerer",
        "rag_count": len(rag),
        "web_count": len(web),
        "citations_count": len(citations)
    })
    
    return state
