from __future__ import annotations
import json, os
from urllib.request import Request, urlopen

class Provider:
    name = "mock"
    def complete(self, messages, tools=None): raise NotImplementedError

class OpenAICompatibleProvider(Provider):
    def __init__(self):
        self.key = os.getenv("LLM_API_KEY") or os.getenv("MOONSHOT_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        self.base = (os.getenv("LLM_BASE_URL") or "https://api.moonshot.cn/v1").rstrip("/")
        self.model = os.getenv("LLM_MODEL") or "kimi-k2.5"
        self.name = "cloud:" + self.model
    def complete(self, messages, tools=None):
        if not self.key: raise RuntimeError("no cloud LLM key configured")
        body = {"model": self.model, "messages": messages, "temperature": 0}
        if tools: body.update({"tools": tools, "tool_choice": "auto"})
        req = Request(self.base + "/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        with urlopen(req, timeout=30) as res: return json.loads(res.read())["choices"][0]["message"]

class MockProvider(Provider):
    name = "deterministic-mock"
    def complete(self, messages, tools=None): return {"role": "assistant", "content": "Fallback mode: use the verified research tools and cite their outputs."}

def get_provider(): return OpenAICompatibleProvider()
