# 1. Install Ollama: https://github.com/ollama/ollama
# 2. Run `ollama serve` to start the server
# 3. Pull a model: `ollama pull llama3.1:8b` (4.9GB) or official Qwen3.8-27B:
#    uv run python -m browser_use.llm.qwen38_27b start


from browser_use import Agent
from browser_use.llm.qwen38_27b import chat_qwen38_27b

llm = chat_qwen38_27b()

Agent('find the founders of browser-use', llm=llm).run_sync()
