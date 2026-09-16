# 1. Install Ollama: https://github.com/ollama/ollama
# 2. Run `ollama serve` to start the server
# 3. Pull a model, e.g. `ollama pull llama3.1:8b` (4.9GB)
#    For Qwen3.8-27B quants: `./bin/setup-qwen38-ollama.sh`


from browser_use import Agent, ChatOllama

llm = ChatOllama(model='llama3.1:8b')

Agent('find the founders of browser-use', llm=llm).run_sync()
