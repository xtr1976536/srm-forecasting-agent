from __future__ import annotations
import json, re, html
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from .schemas import Evidence

def _fetch(url, limit=120000):
    p=urlparse(url)
    if p.scheme not in ("http", "https"): raise ValueError("only http(s) URLs are allowed")
    req=Request(url, headers={"User-Agent":"ResearchAgent/1.0"})
    with urlopen(req, timeout=20) as r: return r.read(limit).decode("utf-8", "replace")

def web_search(query, limit=5):
    # Public DuckDuckGo HTML endpoint; no key is required. Search output is untrusted evidence.
    html=_fetch("https://html.duckduckgo.com/html/?q="+quote(query), 200000)
    out=[]
    for href,title,snippet in re.findall(r'class="result__a" href="(.*?)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',html,re.S)[:limit]:
        clean=lambda s: html.unescape(re.sub("<.*?>", "", s)).strip()
        out.append(Evidence(clean(title), href, "web", clean(snippet), reliability="medium"))
    return out

def web_open(url):
    content=_fetch(url)
    text=html.unescape(re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", content, flags=re.S))
    return Evidence(url, url, "web", re.sub(r"\s+", " ", text)[:500], re.sub(r"\s+", " ", text)[:120000], "medium")

def paper_search(query, limit=5):
    data=json.loads(_fetch("https://api.openalex.org/works?search="+quote(query)+"&per-page="+str(limit)))
    return [Evidence(x.get("title", ""), x.get("doi") or x.get("id", ""), "paper", x.get("publication_year", "") and str(x["publication_year"]), reliability="high") for x in data.get("results", [])]

def github_readme(owner, repo):
    url=f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"
    return Evidence(f"{owner}/{repo} README", url, "github", content=_fetch(url), reliability="high")

def teacher_fit():
    return [{"name":"Yongqi Zhang","focus":"RAG, multimodal reasoning, LLM agents, AI for X"},{"name":"Xiusi Chen","focus":"LLM reasoning, tool use, alignment, structured decision making"},{"name":"Hao Liu","focus":"agentic data science, grounded agents, dynamic world models"},{"name":"Liang Zhang","focus":"general LLM intelligence, financial intelligence, applied AI"},{"name":"Zhenhailong Wang","focus":"predictive thinking, world action models, scientific discovery agents"}]
