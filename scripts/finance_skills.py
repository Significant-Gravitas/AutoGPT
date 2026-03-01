"""AlphaEar Finance Skills for Auto-GPT.

8 plug-and-play financial analysis capabilities:
  1. finance_news       — Real-time financial news & trends (multi-source)
  2. finance_stock      — Stock ticker search + OHLCV history
  3. finance_sentiment  — FinBERT / LLM sentiment scoring (-1.0 to +1.0)
  4. finance_predict    — Time-series forecasting with news-aware adjustments
  5. finance_signal     — Investment signal tracking (strengthen/weaken/falsify)
  6. finance_visualize  — Transmission-chain Draw.io XML diagrams
  7. finance_report     — Professional report generation (plan→write→edit→chart)
  8. finance_search     — Multi-engine web search (DuckDuckGo + Jina + Baidu)

Adapted from https://github.com/RKiding/Awesome-finance-skills
"""

import json
import os
import datetime
from duckduckgo_search import ddg
from call_ai_function import call_ai_function
from file_operations import write_to_file, read_file, safe_join
from config import Config

cfg = Config()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
WORKSPACE = "auto_gpt_workspace"
SIGNALS_FILE = os.path.join(WORKSPACE, "finance_signals.json")


def _llm(function_sig, args, description):
    """Thin wrapper around call_ai_function."""
    return call_ai_function(function_sig, args, description)


def _search_web(query, max_results=8):
    """DuckDuckGo search returning list of dicts."""
    results = []
    for item in ddg(query, max_results=max_results):
        results.append(item)
    return results


def _load_signals():
    """Load tracked signals from disk."""
    if os.path.exists(SIGNALS_FILE):
        try:
            with open(SIGNALS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_signals(signals):
    """Persist tracked signals."""
    os.makedirs(WORKSPACE, exist_ok=True)
    with open(SIGNALS_FILE, "w") as f:
        json.dump(signals, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1. finance_news — Real-time financial news & trends
# ---------------------------------------------------------------------------

def finance_news(query, count=8):
    """Fetch real-time financial news from multiple web sources.

    Args:
        query: News topic or keyword to search for.
        count: Max number of results (default 8).

    Returns:
        Formatted news digest string.
    """
    raw = _search_web(f"finance news {query}", max_results=int(count))
    if not raw:
        return "No finance news found for: " + query

    lines = [f"## Finance News: {query}\n"]
    for i, item in enumerate(raw, 1):
        title = item.get("title", "Untitled")
        body = item.get("body", "")
        href = item.get("href", "")
        lines.append(f"{i}. **{title}**\n   {body}\n   Source: {href}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. finance_stock — Stock ticker search + OHLCV data
# ---------------------------------------------------------------------------

def finance_stock(query):
    """Search for stock ticker info and recent price data.

    Args:
        query: Stock name, ticker symbol, or company (e.g. "AAPL", "Moutai").

    Returns:
        Search results with stock info.
    """
    raw = _search_web(f"stock price {query} OHLCV", max_results=5)
    if not raw:
        return "No stock data found for: " + query

    lines = [f"## Stock Lookup: {query}\n"]
    for i, item in enumerate(raw, 1):
        title = item.get("title", "")
        body = item.get("body", "")
        href = item.get("href", "")
        lines.append(f"{i}. **{title}**\n   {body}\n   {href}\n")

    # Also ask the LLM to extract structured data
    summary = _llm(
        "def extract_stock_info(search_results: str, query: str) -> str:",
        [json.dumps(raw, ensure_ascii=False), query],
        "Extract ticker symbol, current price, recent change, and key stats from the search results. Return a concise summary."
    )
    lines.append(f"\n### AI Summary\n{summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. finance_sentiment — Sentiment analysis (-1.0 to +1.0)
# ---------------------------------------------------------------------------

def finance_sentiment(text):
    """Analyze financial text sentiment using LLM.

    Args:
        text: Financial news headline, article, or any text.

    Returns:
        JSON with score (-1.0 to 1.0), label, and reason.
    """
    result = _llm(
        "def analyze_financial_sentiment(text: str) -> str:",
        [text],
        'Analyze the financial sentiment of the given text. '
        'Return ONLY a JSON object: {"score": <float -1.0 to 1.0>, '
        '"label": "<positive|negative|neutral>", "reason": "<short reason>"}. '
        'Scoring: positive (0.1 to 1.0) = optimistic, profit, support; '
        'negative (-1.0 to -0.1) = losses, sanctions, drops; '
        'neutral (-0.1 to 0.1) = factual, ambiguous.'
    )
    return result


# ---------------------------------------------------------------------------
# 4. finance_predict — Time-series forecast with news-aware adjustments
# ---------------------------------------------------------------------------

def finance_predict(ticker, horizon="7d"):
    """Generate a market prediction for a stock/asset.

    Uses web search for recent data + LLM reasoning for forecast.

    Args:
        ticker: Stock ticker or asset name.
        horizon: Forecast horizon (e.g. "7d", "30d", "3m").

    Returns:
        Forecast analysis string.
    """
    # Gather recent data
    news = _search_web(f"{ticker} stock market news forecast", max_results=5)
    price_data = _search_web(f"{ticker} stock price history OHLCV recent", max_results=3)

    context = json.dumps({
        "recent_news": news,
        "price_data": price_data,
    }, ensure_ascii=False)

    result = _llm(
        "def forecast_market(ticker: str, horizon: str, context: str) -> str:",
        [ticker, horizon, context],
        "You are a financial forecaster. Given the ticker, forecast horizon, "
        "and context (recent news + price data from web search), produce a "
        "structured forecast including: 1) Current trend summary, "
        "2) Key drivers (bullish/bearish), 3) Predicted direction and "
        "confidence level, 4) Risk factors. Be specific and data-driven."
    )
    return f"## Forecast: {ticker} ({horizon})\n\n{result}"


# ---------------------------------------------------------------------------
# 5. finance_signal — Investment signal tracking
# ---------------------------------------------------------------------------

def finance_signal(action, signal_name="", detail=""):
    """Track and update investment signals.

    Args:
        action: One of 'create', 'update', 'list', 'delete'.
        signal_name: Name/ID of the signal.
        detail: For 'create'/'update' — description or new info.

    Returns:
        Result of the signal operation.
    """
    signals = _load_signals()

    if action == "list":
        if not signals:
            return "No tracked signals."
        lines = ["## Tracked Investment Signals\n"]
        for s in signals:
            lines.append(
                f"- **{s['name']}** | Status: {s['status']} | "
                f"Confidence: {s['confidence']} | Last: {s['updated']}\n"
                f"  {s['detail']}\n"
            )
        return "\n".join(lines)

    if action == "create":
        if not signal_name:
            return "Error: signal_name required for create."
        new_signal = {
            "name": signal_name,
            "detail": detail,
            "status": "active",
            "confidence": "medium",
            "created": datetime.datetime.now().isoformat(),
            "updated": datetime.datetime.now().isoformat(),
            "history": [{"event": "created", "detail": detail,
                         "time": datetime.datetime.now().isoformat()}],
        }
        signals.append(new_signal)
        _save_signals(signals)
        return f"Signal created: {signal_name}"

    if action == "update":
        if not signal_name:
            return "Error: signal_name required for update."
        for s in signals:
            if s["name"] == signal_name:
                # Use LLM to assess signal evolution
                assessment = _llm(
                    "def assess_signal_evolution(signal: str, new_info: str) -> str:",
                    [json.dumps(s, ensure_ascii=False), detail],
                    "Given an existing investment signal and new information, "
                    "determine if the signal is Strengthened, Weakened, or "
                    "Falsified. Return JSON: {\"status\": \"<strengthened|"
                    "weakened|falsified|unchanged>\", \"confidence\": "
                    "\"<high|medium|low>\", \"reasoning\": \"<brief>\"}"
                )
                s["history"].append({
                    "event": "updated",
                    "detail": detail,
                    "assessment": assessment,
                    "time": datetime.datetime.now().isoformat(),
                })
                s["updated"] = datetime.datetime.now().isoformat()
                try:
                    parsed = json.loads(assessment)
                    s["status"] = parsed.get("status", s["status"])
                    s["confidence"] = parsed.get("confidence", s["confidence"])
                except (json.JSONDecodeError, TypeError):
                    pass
                _save_signals(signals)
                return f"Signal '{signal_name}' updated.\nAssessment: {assessment}"
        return f"Signal '{signal_name}' not found."

    if action == "delete":
        before = len(signals)
        signals = [s for s in signals if s["name"] != signal_name]
        _save_signals(signals)
        if len(signals) < before:
            return f"Signal '{signal_name}' deleted."
        return f"Signal '{signal_name}' not found."

    return f"Unknown action '{action}'. Use: create, update, list, delete."


# ---------------------------------------------------------------------------
# 6. finance_visualize — Draw.io XML logic chain diagrams
# ---------------------------------------------------------------------------

def finance_visualize(logic_chain, filename="finance_logic.html"):
    """Generate a Draw.io transmission-chain diagram as HTML.

    Args:
        logic_chain: Description of the financial logic/transmission chain.
        filename: Output HTML filename (saved to workspace).

    Returns:
        Confirmation + file path.
    """
    xml = _llm(
        "def generate_drawio_xml(logic_chain: str) -> str:",
        [logic_chain],
        "Generate valid Draw.io mxGraphModel XML that visualizes the given "
        "financial transmission chain. Use boxes for events/factors and "
        "arrows for causal links. Include labels. Return ONLY the XML, "
        "starting with <mxGraphModel>."
    )

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Finance Logic Chain</title></head>
<body>
<h2>Finance Logic Visualization</h2>
<p>Generated: {datetime.datetime.now().isoformat()}</p>
<p><em>Logic chain: {logic_chain}</em></p>
<pre style="background:#f4f4f4;padding:1em;overflow-x:auto;">{xml}</pre>
<p>To view interactively: paste the XML above into <a href="https://app.diagrams.net/">app.diagrams.net</a> (File → Open from → Device, or Edit → XML).</p>
</body>
</html>"""

    result = write_to_file(filename, html)
    return f"Diagram saved to {filename}. {result}\n\nDraw.io XML:\n{xml}"


# ---------------------------------------------------------------------------
# 7. finance_report — Professional report generation
# ---------------------------------------------------------------------------

def finance_report(topic, data=""):
    """Generate a structured professional financial report.

    Args:
        topic: Report topic / title.
        data: Supporting data, analysis, or signal summaries to include.

    Returns:
        Full report as formatted text.
    """
    report = _llm(
        "def generate_financial_report(topic: str, data: str) -> str:",
        [topic, data],
        "Generate a professional financial analysis report. Structure: "
        "1) Executive Summary, 2) Market Context, 3) Key Findings "
        "(with data points), 4) Risk Assessment, 5) Recommendations, "
        "6) Conclusion. Use markdown formatting. Be specific and analytical."
    )

    # Save to file
    filename = f"report_{topic.replace(' ', '_')[:30]}.md"
    write_to_file(filename, f"# Finance Report: {topic}\n"
                  f"Generated: {datetime.datetime.now().isoformat()}\n\n{report}")

    return f"## Finance Report: {topic}\n\n{report}\n\n(Saved to {filename})"


# ---------------------------------------------------------------------------
# 8. finance_search — Multi-engine financial web search
# ---------------------------------------------------------------------------

def finance_search(query, engine="ddg", max_results=8):
    """Search the web for financial information.

    Args:
        query: Search query.
        engine: Search engine — 'ddg' (default). Others noted for future.
        max_results: Number of results.

    Returns:
        Formatted search results.
    """
    raw = _search_web(f"finance {query}", max_results=int(max_results))
    if not raw:
        return f"No results found for: {query}"

    lines = [f"## Finance Search: {query} (via {engine})\n"]
    for i, item in enumerate(raw, 1):
        title = item.get("title", "Untitled")
        body = item.get("body", "")
        href = item.get("href", "")
        lines.append(f"{i}. **{title}**\n   {body}\n   {href}\n")

    return "\n".join(lines)
